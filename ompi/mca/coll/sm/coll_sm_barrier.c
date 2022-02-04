/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2021-2022 Google, LLC. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/** @file */

#include "ompi_config.h"

#include "ompi/constants.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "opal/sys/atomic.h"
#include "coll_sm.h"

/**
 * Shared memory barrier.
 *
 * Tree-based algorithm for a barrier: a fan in to rank 0 followed by
 * a fan out using the barrier segments in the shared memory area.
 *
 * There are 2 sets of barrier buffers -- since there can only be, at
 * most, 2 outstanding barriers at any time, there is no need for more
 * than this.  The generalized in-use flags, control, and data
 * segments are not used.
 *
 * The general algorithm is for a given process to wait for its N
 * children to fan in by monitoring a uint32_t in its barrier "in"
 * buffer.  When this value reaches N (i.e., each of the children have
 * atomically incremented the value), then the process atomically
 * increases the uint32_t in its parent's "in" buffer.  Then the
 * process waits for the parent to set a "1" in the process' "out"
 * buffer.  Once this happens, the process writes a "1" in each of its
 * children's "out" buffers, and returns.
 *
 * There's corner cases, of course, such as the root that has no
 * parent, and the leaves that have no children.  But that's the
 * general idea.
 */
static void mca_coll_sm_barrier_intra_twoproc(struct ompi_communicator_t *comm,
					      mca_coll_sm_module_t *sm_module,
					      mca_coll_sm_comm_t *data,
					      int buffer_set, int bindex,
					      const mca_coll_sm_tree_t *tree)
{
    uint32_t *in = mca_coll_sm_barrier_control(data, ompi_comm_rank(comm), 0, buffer_set, bindex);
    int other_rank = (tree->my_tree_rank == 0) ? 1 : 0;
    uint32_t *out = mca_coll_sm_barrier_control(data, tree->nodes[other_rank].mcstn_id, 0, buffer_set, bindex);

    *out = 1;
    mca_coll_sm_spin_until_equal(in, 1);
    *in = 0;
}

static void mca_coll_sm_barrier_intra_flat(struct ompi_communicator_t *comm,
					   mca_coll_sm_module_t *sm_module,
					   mca_coll_sm_comm_t *data,
					   int buffer_set, int bindex,
					   const mca_coll_sm_tree_t *tree)
{
  for (int i = 0 ; i < tree->node_count ; ++i) {
    opal_atomic_int32_t *value = (opal_atomic_int32_t *) mca_coll_sm_barrier_control(data, tree->nodes[i].mcstn_id, 0, buffer_set, bindex);
    opal_atomic_fetch_add_32(value, 1);
  }
  opal_atomic_int32_t *out = (opal_atomic_int32_t *) mca_coll_sm_barrier_control(data, tree->nodes[tree->my_tree_rank].mcstn_id, 0, buffer_set, bindex);
  mca_coll_sm_spin_until_equal((uint32_t *) out, tree->node_count);
  *out = 0;
}

static inline void mca_coll_sm_barrier_intra_tree_gather_phase(struct ompi_communicator_t *comm,
							       mca_coll_sm_module_t *sm_module,
							       mca_coll_sm_comm_t *data,
							       int buffer_set, int bindex,
							       const mca_coll_sm_tree_t *tree)
{
    const mca_coll_sm_tree_node_t *my_tree_node = tree->nodes + tree->my_tree_rank;
    int num_children = my_tree_node->mcstn_num_children;
    volatile uint32_t *in = mca_coll_sm_barrier_control(data, my_tree_node->mcstn_id, MCA_COLL_SM_BARRIER_DIRECTION_IN, buffer_set, bindex);
    volatile uint32_t *parent_in = my_tree_node->mcstn_parent ? mca_coll_sm_barrier_control(data, my_tree_node->mcstn_parent->mcstn_id, MCA_COLL_SM_BARRIER_DIRECTION_IN, buffer_set, bindex) : NULL;

    /* Wait for my children to write to my *in* buffer */
    if (0 != num_children) {
	mca_coll_sm_spin_until_equal(in, num_children);
	*in = 0;
    }

    /* Send to my parent and wait for a response (don't poll on
       parent's out buffer -- that would cause a lot of network
       traffic / contention / faults / etc.  Instead, children poll on
       local memory and therefore only num_children messages are sent
       across the network [vs. num_children *each* time all the
       children poll] -- i.e., the memory is only being polled by one
       process, and it is only changed *once* by an external
       process) */

    if (NULL != parent_in) {
	/* Signal to the parent that this process has arrived using its in buffer. */
	opal_atomic_fetch_add_32 ((opal_atomic_int32_t *)parent_in, 1);
    }
}

static inline void mca_coll_sm_barrier_intra_tree_bcast_phase(struct ompi_communicator_t *comm,
							      mca_coll_sm_module_t *sm_module,
							      mca_coll_sm_comm_t *data,
							      int buffer_set, int bindex,
							      const mca_coll_sm_tree_t *tree)
{
    int rank = ompi_comm_rank(comm);
    const mca_coll_sm_tree_node_t *my_tree_node = tree->nodes + tree->my_tree_rank;
    int num_children = my_tree_node->mcstn_num_children;
    volatile uint32_t *out = mca_coll_sm_barrier_control(data, rank, MCA_COLL_SM_BARRIER_DIRECTION_OUT, buffer_set, bindex);

    if (0 != rank) {
	mca_coll_sm_spin_until_equal(out, 1);
        *out = 0;
    }

    /* Send to my children */
    for (int i = 0; i < num_children; ++i) {
      uint32_t *child_out = mca_coll_sm_barrier_control(data, my_tree_node->mcstn_children[i]->mcstn_id, MCA_COLL_SM_BARRIER_DIRECTION_OUT, buffer_set, bindex);
      *child_out = 1;
    }

    /* All done!  End state of the control segment should be zeroed */
}

static void mca_coll_sm_barrier_intra_tree(struct ompi_communicator_t *comm,
					   mca_coll_sm_module_t *sm_module,
					   mca_coll_sm_comm_t *data,
					   int buffer_set, int bindex,
					   const mca_coll_sm_tree_t *tree)
{
    int rank = ompi_comm_rank(comm);
    volatile uint32_t *me_in, *me_out;
    const mca_coll_sm_tree_node_t *my_tree_node = tree->nodes + tree->my_tree_rank;
    int num_children = my_tree_node->mcstn_num_children;

    if (-1 == tree->my_tree_rank) {
	return;
    }

    if (2 == tree->node_count) {
      mca_coll_sm_barrier_intra_twoproc(comm, sm_module, data, buffer_set, bindex, tree);
      return;
    }

    mca_coll_sm_barrier_intra_tree_gather_phase(comm, sm_module, data, buffer_set, bindex, tree);
    mca_coll_sm_barrier_intra_tree_bcast_phase(comm, sm_module, data, buffer_set, bindex, tree);
}

int mca_coll_sm_barrier_intra(struct ompi_communicator_t *comm,
                              mca_coll_base_module_t *module)
{
    mca_coll_sm_module_t *sm_module = (mca_coll_sm_module_t*) module;

    /* Lazily enable the module the first time we invoke a collective
       on it */
    if (!sm_module->enabled) {
        int ret;
        if (OMPI_SUCCESS != (ret = ompi_coll_sm_lazy_enable(module, comm))) {
            return ret;
        }
    }

    mca_coll_sm_comm_t *data = sm_module->sm_comm_data;
    int buffer_set = (data->mcb_barrier_count++) & 1;

    if (NULL == data->mcb_inter_numa_tree) {
	mca_coll_sm_barrier_intra_tree (comm, sm_module, data, buffer_set, /*bindex=*/0, data->mcb_intra_numa_tree);
	return OMPI_SUCCESS;
    }

    mca_coll_sm_barrier_intra_tree_gather_phase(comm, sm_module, data, buffer_set, /*bindex=*/0, data->mcb_inter_numa_tree);
    mca_coll_sm_barrier_intra_tree(comm, sm_module, data, buffer_set, /*bindex=*/1, data->mcb_inter_numa_tree);
    mca_coll_sm_barrier_intra_tree_bcast_phase(comm, sm_module, data, buffer_set, /*bindex=*/0, data->mcb_inter_numa_tree);

    return OMPI_SUCCESS;
}
