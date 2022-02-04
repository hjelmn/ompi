/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009-2013 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include <string.h>

#include "opal/datatype/opal_convertor.h"
#include "opal/sys/atomic.h"
#include "ompi/constants.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/op/op.h"
#include "coll_sm.h"


/*
 * Local functions
 */
static int reduce_inorder(const void *sbuf, void* rbuf, int count,
                          struct ompi_datatype_t *dtype,
                          struct ompi_op_t *op,
                          int root, struct ompi_communicator_t *comm,
                          mca_coll_base_module_t *module);
#define WANT_REDUCE_NO_ORDER 0
#if WANT_REDUCE_NO_ORDER
static int reduce_no_order(const void *sbuf, void* rbuf, int count,
                           struct ompi_datatype_t *dtype,
                           struct ompi_op_t *op,
                           int root, struct ompi_communicator_t *comm,
                           mca_coll_base_module_t *module);
#endif

/*
 * Useful utility routine
 */
#if !defined(min)
static inline int min(int a, int b)
{
    return (a < b) ? a : b;
}
#endif

/**
 * Shared memory reduction.
 *
 * Simply farms out to the associative or non-associative functions.
 */
int mca_coll_sm_reduce_intra(const void *sbuf, void* rbuf, int count,
                             struct ompi_datatype_t *dtype,
                             struct ompi_op_t *op,
                             int root, struct ompi_communicator_t *comm,
                             mca_coll_base_module_t *module)
{
  return OMPI_ERR_NOT_IMPLEMENTED;
}
