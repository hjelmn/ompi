/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/** @file */

#include "ompi_config.h"

#include <string.h>

#include "opal/datatype/opal_convertor.h"
#include "ompi/constants.h"
#include "ompi/communicator/communicator.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/coll/coll.h"
#include "opal/sys/atomic.h"
#include "coll_sm.h"

/**
 * Shared memory broadcast.
 *
 * For the root, the general algorithm is to wait for a set of
 * segments to become available.  Once it is, the root claims the set
 * by writing the current operation number and the number of processes
 * using the set to the flag.  The root then loops over the set of
 * segments; for each segment, it copies a fragment of the user's
 * buffer into the shared data segment and then writes the data size
 * into its childrens' control buffers.  The process is repeated until
 * all fragments have been written.
 *
 * For non-roots, for each set of buffers, they wait until the current
 * operation number appears in the in-use flag (i.e., written by the
 * root).  Then for each segment, they wait for a nonzero to appear
 * into their control buffers.  If they have children, they copy the
 * data from their parent's shared data segment into their shared data
 * segment, and write the data size into each of their childrens'
 * control buffers.  They then copy the data from their shared [local]
 * data segment into the user's output buffer.  The process is
 * repeated until all fragments have been received.  If they do not
 * have children, they copy the data directly from the parent's shared
 * data segment into the user's output buffer.
 */
int mca_coll_sm_bcast_intra(void *buff, int count,
                            struct ompi_datatype_t *datatype, int root,
                            struct ompi_communicator_t *comm,
                            mca_coll_base_module_t *module)
{
  return OMPI_ERR_NOT_IMPLEMENTED;
}
