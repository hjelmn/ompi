/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2009-2013 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2010-2012 Los Alamos National Security, LLC.
 *                         All rights reserved.
 * Copyright (c) 2014-2015 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2015-2019 Intel, Inc.  All rights reserved.
 * Copyright (c) 2018      Amazon.com, Inc. or its affiliates.  All Rights reserved.
 * Copyright (c) 2021-2022 Google, LLC. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 *
 * Warning: this is not for the faint of heart -- don't even bother
 * reading this source code if you don't have a strong understanding
 * of nested data structures and pointer math (remember that
 * associativity and order of C operations is *critical* in terms of
 * pointer math!).
 */

#include "ompi_config.h"

#include <stdio.h>
#include <string.h>
#ifdef HAVE_SCHED_H
#include <sched.h>
#endif
#include <sys/types.h>
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif  /* HAVE_SYS_MMAN_H */
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif  /* HAVE_UNISTD_H */

#include "mpi.h"
#include "opal_stdint.h"
#include "opal/mca/hwloc/base/base.h"
#include "opal/util/os_path.h"
#include "opal/util/minmax.h"
#include "opal/util/printf.h"

#include "ompi/runtime/ompi_rte.h"
#include "ompi/communicator/communicator.h"
#include "ompi/group/group.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/proc/proc.h"
#include "coll_sm.h"

#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"

/*
 * Local functions
 */
static int sm_module_enable(mca_coll_base_module_t *module,
                          struct ompi_communicator_t *comm);
static int bootstrap_comm(ompi_communicator_t *comm,
                          mca_coll_sm_module_t *module);
static int mca_coll_sm_module_disable(mca_coll_base_module_t *module,
                          struct ompi_communicator_t *comm);
static int mca_coll_sm_generate_numa_trees(mca_coll_sm_module_t *sm_module, ompi_communicator_t *comm, int degree);
static void mca_coll_sm_print_tree(const mca_coll_sm_tree_t *tree);


/*
 * Module constructor
 */
static void mca_coll_sm_module_construct(mca_coll_sm_module_t *module)
{
    module->enabled = false;
    module->sm_comm_data = NULL;
    module->previous_reduce = NULL;
    module->previous_reduce_module = NULL;
    module->super.coll_module_disable = mca_coll_sm_module_disable;
}

/*
 * Module destructor
 */
static void mca_coll_sm_module_destruct(mca_coll_sm_module_t *module)
{
    mca_coll_sm_comm_t *c = module->sm_comm_data;

    if (NULL != c) {
        /* Munmap the per-communicator shmem data segment */
        if (NULL != c->sm_segment_base) {
            opal_shmem_segment_detach(&c->sm_segment);
        }

        free(c);
    }

    /* It should always be non-NULL, but just in case */
    if (NULL != module->previous_reduce_module) {
        OBJ_RELEASE(module->previous_reduce_module);
    }

    module->enabled = false;
}

/*
 * Module disable
 */
static int mca_coll_sm_module_disable(mca_coll_base_module_t *module, struct ompi_communicator_t *comm)
{
    mca_coll_sm_module_t *sm_module = (mca_coll_sm_module_t*) module;
    if (NULL != sm_module->previous_reduce_module) {
	sm_module->previous_reduce = NULL;
        OBJ_RELEASE(sm_module->previous_reduce_module);
	sm_module->previous_reduce_module = NULL;
    }
    return OMPI_SUCCESS;
}


OBJ_CLASS_INSTANCE(mca_coll_sm_module_t,
                   mca_coll_base_module_t,
                   mca_coll_sm_module_construct,
                   mca_coll_sm_module_destruct);

/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.  This function is invoked exactly
 * once.
 */
int mca_coll_sm_init_query(bool enable_progress_threads,
                           bool enable_mpi_threads)
{
    /* if no session directory was created, then we cannot be used */
    if (NULL == ompi_process_info.job_session_dir) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    /* Don't do much here because we don't really want to allocate any
       shared memory until this component is selected to be used. */
    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:sm:init_query: pick me! pick me!");
    return OMPI_SUCCESS;
}


/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
mca_coll_sm_comm_query(struct ompi_communicator_t *comm, int *priority)
{
    mca_coll_sm_module_t *sm_module;

    /* If we're intercomm, or if there's only one process in the
       communicator, or if not all the processes in the communicator
       are not on this node, then we don't want to run */
    if (OMPI_COMM_IS_INTER(comm) || 1 == ompi_comm_size(comm) || ompi_group_have_remote_peers (comm->c_local_group)) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:sm:comm_query (%s/%s): intercomm, comm is too small, or not all peers local; disqualifying myself",
			    ompi_comm_print_cid (comm), comm->c_name);
	return NULL;
    }

    /* Get the priority level attached to this module. If priority is less
     * than or equal to 0, then the module is unavailable. */
    *priority = mca_coll_sm_component.sm_priority;
    if (mca_coll_sm_component.sm_priority <= 0) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:sm:comm_query (%s/%s): priority too low; disqualifying myself",
			    ompi_comm_print_cid (comm), comm->c_name);
	return NULL;
    }

    sm_module = OBJ_NEW(mca_coll_sm_module_t);
    if (NULL == sm_module) {
        return NULL;
    }

    /* All is good -- return a module */
    sm_module->super.coll_module_enable = sm_module_enable;
    sm_module->super.coll_allgather  = NULL;
    sm_module->super.coll_allgatherv = NULL;
    sm_module->super.coll_allreduce  = NULL;
    sm_module->super.coll_alltoall   = NULL;
    sm_module->super.coll_alltoallv  = NULL;
    sm_module->super.coll_alltoallw  = NULL;
    sm_module->super.coll_barrier    = mca_coll_sm_barrier_intra;
    sm_module->super.coll_bcast      = NULL;
    sm_module->super.coll_exscan     = NULL;
    sm_module->super.coll_gather     = NULL;
    sm_module->super.coll_gatherv    = NULL;
    sm_module->super.coll_reduce     = NULL;
    sm_module->super.coll_reduce_scatter = NULL;
    sm_module->super.coll_scan       = NULL;
    sm_module->super.coll_scatter    = NULL;
    sm_module->super.coll_scatterv   = NULL;

#if 0
    /* Save previous component's reduce information */
    sm_module->previous_reduce = comm->c_coll->coll_reduce;
    sm_module->previous_reduce_module = comm->c_coll->coll_reduce_module;
    OBJ_RETAIN(comm->c_coll->coll_reduce_module);
#endif

    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:sm:comm_query (%s/%s): pick me! pick me!",
                        ompi_comm_print_cid (comm), comm->c_name);
    return &(sm_module->super);
}


/*
 * Init module on the communicator
 */
static int sm_module_enable(mca_coll_base_module_t *module,
                            struct ompi_communicator_t *comm)
{
    if (NULL == comm->c_coll->coll_reduce ||
        NULL == comm->c_coll->coll_reduce_module) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:sm:enable (%s/%s): no underlying reduce; disqualifying myself",
			    ompi_comm_print_cid (comm), comm->c_name);
        return OMPI_ERROR;
    }

    /* We do everything lazily in ompi_coll_sm_enable() */
    return OMPI_SUCCESS;
}

static int ompi_coll_sm_setup_maffinity(mca_coll_sm_module_t *sm_module, ompi_communicator_t *comm, unsigned char *base)
{
    mca_coll_sm_comm_t *data = sm_module->sm_comm_data;
    mca_coll_sm_component_t *component = &mca_coll_sm_component;
    int rank = ompi_comm_rank (comm);
    int size = ompi_comm_size(comm);
 
   /* Get some space to setup memory affinity (just easier to try to
       alloc here to handle the error case) */
    opal_hwloc_base_memory_segment_t *maffinity = (opal_hwloc_base_memory_segment_t*)
        malloc(sizeof(opal_hwloc_base_memory_segment_t) *
               component->sm_comm_num_segments * 3);
    if (NULL == maffinity) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:sm:enable (%s/%s): malloc failed (1)",
                            ompi_comm_print_cid (comm), comm->c_name);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* All things being equal, if we're rank 0, then make the in-use
       flags be local (memory affinity).  Then zero them all out so
       that they're marked as unused. */
    int j = 0;
    if (0 == rank) {
        maffinity[j].mbs_start_addr = base;
        maffinity[j].mbs_len = component->sm_control_size *
            component->sm_comm_num_in_use_flags;
        /* Set the op counts to 1 (actually any nonzero value will do)
           so that the first time children/leaf processes come
           through, they don't see a value of 0 and think that the
           root/parent has already set the count to their op number
           (i.e., 0 is the first op count value). */
        for (int i = 0; i < mca_coll_sm_component.sm_comm_num_in_use_flags; ++i) {
            ((mca_coll_sm_in_use_flag_t *)base)[i].mcsiuf_operation_count = 1;
            ((mca_coll_sm_in_use_flag_t *)base)[i].mcsiuf_num_procs_using = 0;
        }
        ++j;
    }

    /* Next, setup pointers to the control and data portions of the
       segments, as well as to the relevant in-use flags. */
    base += (component->sm_comm_num_in_use_flags * component->sm_control_size);
    size_t control_size = size * component->sm_control_size;
    size_t frag_size = size * component->sm_fragment_size;
    for (int i = 0; i < component->sm_comm_num_segments; ++i) {
        data->mcb_data_index[i].mcbmi_control = (uint32_t*)
            (base + (i * (control_size + frag_size)));
        data->mcb_data_index[i].mcbmi_data =
            (((char*) data->mcb_data_index[i].mcbmi_control) +
             control_size);

        /* Memory affinity: control */
        maffinity[j].mbs_len = component->sm_control_size;
        maffinity[j].mbs_start_addr = (void *)
            (((char*) data->mcb_data_index[i].mcbmi_control) +
             (rank * component->sm_control_size));
        ++j;

        /* Memory affinity: data */
        maffinity[j].mbs_len = component->sm_fragment_size;
        maffinity[j].mbs_start_addr =
            ((char*) data->mcb_data_index[i].mcbmi_data) +
            (rank * component->sm_control_size);
        ++j;
    }

    /* Setup memory affinity so that the pages that belong to this
       process are local to this process */
    opal_hwloc_base_memory_set(maffinity, j);
    free(maffinity);

        return OMPI_SUCCESS;
}

int ompi_coll_sm_lazy_enable(mca_coll_base_module_t *module,
                             struct ompi_communicator_t *comm)
{
    int root, ret;
    int rank = ompi_comm_rank(comm);
    int size = ompi_comm_size(comm);
    mca_coll_sm_module_t *sm_module = (mca_coll_sm_module_t*) module;
    mca_coll_sm_comm_t *data = NULL;
    size_t frag_size;
    mca_coll_sm_component_t *component = &mca_coll_sm_component;
    int parent, min_child, num_children;
    unsigned char *base = NULL;

    /* Just make sure we haven't been here already */
    if (sm_module->enabled) {
        return OMPI_SUCCESS;
    }
    sm_module->enabled = true;

    /* Allocate data to hang off the communicator.  The memory we
       alloc will be laid out as follows:

       1. mca_coll_base_comm_t
       2. array of num_segments mca_coll_base_mpool_index_t instances
          (pointed to by the array in 2)
       3. array of ompi_comm_size(comm) mca_coll_sm_tree_node_t
          instances
       4. array of sm_tree_degree pointers to other tree nodes (i.e.,
          this nodes' children) for each instance of
          mca_coll_sm_tree_node_t
    */
    sm_module->sm_comm_data = data = (mca_coll_sm_comm_t*)
        calloc(1, sizeof(mca_coll_sm_comm_t) +
               (component->sm_comm_num_segments *
                sizeof(mca_coll_sm_data_index_t)));
    if (NULL == data) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:sm:enable (%s/%s): malloc failed (2)",
                            ompi_comm_print_cid (comm), comm->c_name);
        return OMPI_ERR_TEMP_OUT_OF_RESOURCE;
    }

    mca_coll_sm_generate_numa_trees(sm_module, comm, component->sm_tree_degree);

    /* Attach to this communicator's shmem data segment */
    if (OMPI_SUCCESS != (ret = bootstrap_comm(comm, sm_module))) {
        free(data);
        sm_module->sm_comm_data = NULL;
        return ret;
    }

    /* Once the communicator is bootstrapped, setup the pointers into
       the per-communicator shmem data segment.  First, setup the
       barrier buffers.  There are 2 sets of barrier buffers (because
       there can never be more than one outstanding barrier occuring
       at any timie).  Setup pointers to my control buffers, my
       parents, and [the beginning of] my children (note that the
       children are contiguous, so having the first pointer and the
       num_children from the mcb_tree data is sufficient). */
    base = data->sm_segment_base;

    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:sm:enable (%s/%s): base=%p",  ompi_comm_print_cid (comm), comm->c_name, base);

    /* Next, setup the pointer to the in-use flags.  The number of
       segments will be an even multiple of the number of in-use
       flags. */
    base += (size * mca_coll_sm_barrier_buffer_size());
    data->mcb_in_use_flags = (mca_coll_sm_in_use_flag_t *) base;

    if (NULL != sm_module->sm_comm_data->mcb_inter_numa_tree) {
        /* more than one NUMA domain is in use. turn on memory affinity */
        ret = ompi_coll_sm_setup_maffinity(sm_module, comm, base);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
            opal_output_verbose(ompi_coll_base_framework.framework_output, MCA_BASE_VERBOSE_WARN, "Could not set up memory affinity for coll/sm module. This may reduce performance.");
        }

        for (int i = 0; i < component->sm_comm_num_segments; ++i) {
            memset((void *) data->mcb_data_index[i].mcbmi_control, 0,
                   component->sm_control_size);
        }
    }

    /* Zero out the control structures that belong to this process */
    uint32_t *my_barrier_control = mca_coll_sm_barrier_control(data, rank, /*direction=*/0, /*buffer_set=*/0, /*bindex=*/0);
    memset (my_barrier_control, 0, mca_coll_sm_barrier_buffer_size());

    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:sm:enable (%s/%s): waiting for peers to attach",
                        ompi_comm_print_cid (comm), comm->c_name);
    /* Wait for everyone in this communicator to attach and setup */
    ompi_coll_base_barrier_intra_recursivedoubling(comm, module);

    if (0 == rank) {
        opal_shmem_unlink (&data->sm_segment);
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:sm:enable (%s/%s): removed mmap file %s",
                            ompi_comm_print_cid (comm), comm->c_name,
                            data->sm_segment.seg_name);
    }


    /* All done */
    opal_output_verbose(MCA_BASE_VERBOSE_COMPONENT, ompi_coll_base_framework.framework_output,
                        "coll:sm:enable (%s/%s): success!",
                        ompi_comm_print_cid (comm), comm->c_name);

    return OMPI_SUCCESS;
}

static int bootstrap_comm(ompi_communicator_t *comm,
                          mca_coll_sm_module_t *module)
{
    mca_coll_sm_component_t *component = &mca_coll_sm_component;
    mca_coll_sm_comm_t *data = module->sm_comm_data;
    int comm_size = ompi_comm_size(comm);
    int num_segments = component->sm_comm_num_segments;
    int num_in_use = component->sm_comm_num_in_use_flags;
    int frag_size = component->sm_fragment_size;
    int ret;
    size_t size;
    ompi_proc_t *proc;

    /* Calculate how much space we need in the per-communicator shmem
       data segment.  There are several values to add:

       - size of the barrier data (2 of these):
           - fan-in data (num_procs * control_size)
           - fan-out data (num_procs * control_size)
       - size of the "in use" buffers:
           - num_in_use_buffers * control_size
       - size of the message fragment area (one for each segment):
           - control (num_procs * control_size)
           - fragment data (num_procs * (frag_size))

       So it's:

           barrier: 2 * control_size + 2 * control_size
           in use:  num_in_use * control_size
           control: num_segments * (num_procs * control_size * 2 +
                                    num_procs * control_size)
           message: num_segments * (num_procs * frag_size)
     */

    size = 4 * component->sm_control_size +
        (num_in_use * component->sm_control_size) +
        (num_segments * (comm_size * component->sm_control_size * 2)) +
        (num_segments * (comm_size * frag_size));
    if (0 == ompi_comm_rank (comm)) {
        char *fullpath;

        /* Job name plus the unique communicator name are sufficient to ensure no collisions between segment files. */
        ret = opal_asprintf (&fullpath, "%s" OPAL_PATH_SEP "coll_sm_segment.%x.%s",
                             component->sm_backing_directory, OMPI_PROC_MY_NAME->jobid, ompi_comm_print_cid(comm));
        if (ret < 0) {
            opal_output_verbose(MCA_BASE_VERBOSE_COMPONENT, ompi_coll_base_framework.framework_output,
                                "coll:sm:enable:bootstrap comm (%s/%s): asprintf failed with code %d",
                                ompi_comm_print_cid (comm), comm->c_name, ret);
            return ret;
        }

        opal_output_verbose(MCA_BASE_VERBOSE_COMPONENT, ompi_coll_base_framework.framework_output,
                            "coll:sm:enable:bootstrap comm (%s/%s): creating %" PRIsize_t " byte segment: %s",
                            ompi_comm_print_cid (comm), comm->c_name, size, fullpath);
        ret = opal_shmem_segment_create (&data->sm_segment, fullpath, size);
        free(fullpath);
        if (OPAL_SUCCESS != ret) {
            opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                "coll:sm:enable:bootstrap comm (%s/%s): opal_shmem_segment_create failed",
				ompi_comm_print_cid (comm), comm->c_name);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }

        data->sm_segment_base = opal_shmem_segment_attach (&data->sm_segment);
        for (int i = 1 ; i < ompi_comm_size (comm) ; ++i) {
            MCA_PML_CALL(send(&data->sm_segment, sizeof (data->sm_segment), MPI_BYTE,
                         i, MCA_COLL_BASE_TAG_BCAST, MCA_PML_BASE_SEND_STANDARD, comm));
        }
    } else {
        opal_shmem_ds_t shmem_ds;
        MCA_PML_CALL(recv(&data->sm_segment, sizeof (data->sm_segment), MPI_BYTE, 0, MCA_COLL_BASE_TAG_BCAST, comm, MPI_STATUS_IGNORE));
        opal_output_verbose(MCA_BASE_VERBOSE_COMPONENT, ompi_coll_base_framework.framework_output,
                            "coll:sm:enable:bootstrap comm (%s/%s): attaching to %" PRIsize_t " byte segment: %s",
                            ompi_comm_print_cid (comm), comm->c_name, size, data->sm_segment.seg_name);

        data->sm_segment_base = opal_shmem_segment_attach (&data->sm_segment);
    }

    return OMPI_SUCCESS;
}


/* TODO -- move this code when complete */

static int mca_coll_sm_get_numa_count(void) {
  int rc = opal_hwloc_base_get_topology();
  if (OPAL_SUCCESS != rc) {
    opal_output_verbose(ompi_coll_base_framework.framework_output, MCA_BASE_VERBOSE_WARN, "mca_coll_sm_get_numa_count: could not get hwloc topology. assuming a single NUMA domain");
    return 1;
  }

  return opal_hwloc_base_get_nbobjs_by_type(opal_hwloc_topology, HWLOC_OBJ_NUMANODE, /*cache_level=*/0, OPAL_HWLOC_AVAILABLE);
}

static int mca_coll_sm_get_my_numa(void) {
  int numa_count = mca_coll_sm_get_numa_count();

  if (2 > numa_count) {
    return 0;
  }

  hwloc_nodeset_t node_set = hwloc_bitmap_alloc();
  hwloc_cpuset_to_nodeset(opal_hwloc_topology, opal_hwloc_my_cpuset, node_set);
  int my_numa = hwloc_bitmap_first(node_set);
  if (hwloc_bitmap_last(node_set) != my_numa) {
    opal_output_verbose(ompi_coll_base_framework.framework_output, MCA_BASE_VERBOSE_WARN, "mca_coll_sm_get_my_numa: process is bound to multiple numa domains. disabling NUMA-aware optimizations");
    my_numa = -1;
  }
  hwloc_bitmap_free(node_set);
  return my_numa;
}

static int *mca_coll_sm_get_numa_mapping_for_comm(ompi_communicator_t *comm) {
  int *mapping;

  mapping = calloc (ompi_comm_size(comm), sizeof (int));
  if (OPAL_UNLIKELY(NULL == mapping)) {
    opal_output_verbose(ompi_coll_base_framework.framework_output, MCA_BASE_VERBOSE_WARN, "mca_coll_sm_get_numa_mapping: could not allocate memory for NUMA mapping");
    return NULL;
  }

  mapping[ompi_comm_rank(comm)] = mca_coll_sm_get_my_numa();

  int rc = comm->c_coll->coll_allgather(MPI_IN_PLACE, 1, MPI_INT, mapping, 1, MPI_INT, comm, comm->c_coll->coll_allgather_module);
  if (OMPI_SUCCESS != rc) {
    opal_output_verbose(ompi_coll_base_framework.framework_output, MCA_BASE_VERBOSE_WARN, "mca_coll_sm_get_numa_mapping: allgather of NUMA info failed. disabling NUMA-aware optimizations");
    free(mapping);
    return NULL;
  }

  return mapping;
}

static mca_coll_sm_tree_t *mca_coll_sm_get_tree(ompi_communicator_t *comm, int *ranks, int rank_count, int degree) {
    if (0 == degree || 0 == rank_count) {
        return NULL;
    }

    unsigned char *tree_buffer = calloc (1, sizeof (mca_coll_sm_tree_t) + rank_count * (sizeof (mca_coll_sm_tree_node_t) + sizeof (mca_coll_sm_tree_node_t *)));

    mca_coll_sm_tree_t *tree = (mca_coll_sm_tree_t *) tree_buffer;
    mca_coll_sm_tree_node_t *root = (mca_coll_sm_tree_node_t *) (tree + 1);
    mca_coll_sm_tree_node_t **children = (mca_coll_sm_tree_node_t **) (root + rank_count);
    int my_rank = ompi_comm_rank(comm);

    tree->my_tree_rank = -1;
    tree->node_count = rank_count;

    for (int level = 0, node_index = 0, nodes_at_level = 1, first_child_index = 1 ; node_index < rank_count ; ++level) { 
        int remaining_processes = rank_count - node_index;

        if (remaining_processes < nodes_at_level) {
            nodes_at_level = remaining_processes;
        }


        int total_left = rank_count - node_index - nodes_at_level;
        int max_nodes_at_next_level = nodes_at_level * degree;
        int nodes_at_next_level = opal_min(max_nodes_at_next_level, total_left);

        int next_level_nodes = nodes_at_next_level;
        
        for (int node = 0, child_count = nodes_at_next_level ; node < nodes_at_level ; ++node, ++node_index) {
            mca_coll_sm_tree_node_t *tree_node = root + node_index;
 
            /* balance the nodes */
            int children_per_node = (nodes_at_next_level + nodes_at_level - node - 1) / (nodes_at_level - node);
            
            if (my_rank == ranks[node_index]) {
                tree->my_tree_rank = node_index;
            }

            tree_node->mcstn_id = ranks[node_index];
            tree_node->mcstn_children = children;
            tree_node->mcstn_num_children = opal_min(nodes_at_next_level, children_per_node);

            nodes_at_next_level -= tree_node->mcstn_num_children;
            children += tree_node->mcstn_num_children;
            for (int child = 0 ; child < tree_node->mcstn_num_children ; ++child, ++first_child_index) {
                tree_node->mcstn_children[child] = root + first_child_index;
                tree_node->mcstn_children[child]->mcstn_parent = tree_node;
            }
        }

        nodes_at_level = next_level_nodes;
    }

    return tree;
}

static int mca_coll_sm_generate_numa_trees(mca_coll_sm_module_t *sm_module, ompi_communicator_t *comm, int degree) {
  int comm_size = ompi_comm_size(comm);
  int comm_rank = ompi_comm_rank(comm);
  int numa_count = mca_coll_sm_get_numa_count();
  int *numa_array = alloca(sizeof(int) * comm_size);
  int *leader_array = alloca(sizeof(int) * numa_count);
  int *numa_counts = alloca(sizeof(int) * numa_count);
  int *mapping = mca_coll_sm_get_numa_mapping_for_comm(comm);
  int leaders = 1;

  if (NULL == mapping || numa_count == 1) {
      /* could not get a mapping or single numa. treat this like the single numa case */
      for (int i = 0 ; i < comm_size ; ++i) {
          numa_array[i] = i;
      }

      sm_module->sm_comm_data->mcb_intra_numa_tree = mca_coll_sm_get_tree(comm, numa_array, comm_size, degree);
      sm_module->sm_comm_data->mcb_inter_numa_tree = NULL;

      return OMPI_SUCCESS;
  }

  memset(numa_counts, 0, sizeof(numa_counts[0]) * numa_count);

  int numa_rank_count = 0, leader_count = 0, my_numa = mapping[comm_rank];
  for (int i = 0 ; i < comm_size ; ++i) {
    if (mapping[i] == my_numa) {
      numa_array[numa_rank_count++] = i;
    }
    if (numa_counts[mapping[i]]++ == 0) {
      leader_array[leader_count++] = i;
    }
  }

  free(mapping);

  if (leader_count > 1) {
      sm_module->sm_comm_data->mcb_inter_numa_tree = mca_coll_sm_get_tree(comm, leader_array, leader_count, degree);
  } else {
      sm_module->sm_comm_data->mcb_inter_numa_tree = NULL;
  }

  sm_module->sm_comm_data->mcb_intra_numa_tree = mca_coll_sm_get_tree(comm, numa_array, numa_rank_count, degree);
  return OMPI_SUCCESS;
}

static void mca_coll_sm_print_tree(const mca_coll_sm_tree_t *tree) {
    if (0 != tree->my_tree_rank) {
        return;
    }

    static int foo=1;
    if (foo==0) {
        return;
    }
    foo=0;

    for (int i = 0 ; i < tree->node_count ; ++i) {
        const mca_coll_sm_tree_node_t *node = tree->nodes + i;
        printf ("Node %d: rank=%d, child_count=%d, parent=%d\n", i, node->mcstn_id, node->mcstn_num_children, node->mcstn_parent ? node->mcstn_parent->mcstn_id : -1);
        printf ("  Children: \n");
        for (int j = 0 ; j < node->mcstn_num_children ; ++j) {
            printf ("    %d\n", node->mcstn_children[j]->mcstn_id);
        }
    }
}
