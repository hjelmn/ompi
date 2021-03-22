/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006      University of Houston. All rights reserved.
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "ompi_config.h"
#include <stdio.h>

#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/group/group.h"
#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"

#if OMPI_BUILD_MPI_PROFILING
#    if OPAL_HAVE_WEAK_SYMBOLS
#        pragma weak MPI_Group_intersection = PMPI_Group_intersection
#    endif
#    define MPI_Group_intersection PMPI_Group_intersection
#endif

static const char FUNC_NAME[] = "MPI_Group_intersection";

int MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *new_group)
{
    int err;

    if (MPI_PARAM_CHECK) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);

        /* verify that groups are valid */
        if ((MPI_GROUP_NULL == group1) || (MPI_GROUP_NULL == group2) || (NULL == group1)
            || (NULL == group2) || (NULL == new_group)) {
            return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_GROUP, FUNC_NAME);
        }
    }

    err = ompi_group_intersection(group1, group2, new_group);
    OMPI_ERRHANDLER_NOHANDLE_RETURN(err, err, FUNC_NAME);
}
