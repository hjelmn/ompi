/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2020      Google, LLC. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_MCA_BASE_ALIAS_H
#define OPAL_MCA_BASE_ALIAS_H

#include "opal_config.h"
#include "opal/class/opal_list.h"

BEGIN_C_DECLS

enum {
      MCA_BASE_ALIAS_FLAG_NONE       = 0,
      MCA_BASE_ALIAS_FLAG_DEPRECATED = 1,
};

struct mca_base_alias_item_t {
    opal_list_item_t super;
    char *component_alias;
    uint32_t alias_flags;
};

typedef struct mca_base_alias_item_t mca_base_alias_item_t;

OBJ_CLASS_DECLARATION(mca_base_alias_item_t);

struct mca_base_alias_t {
    opal_object_t super;
    opal_list_t component_aliases;
};

typedef struct mca_base_alias_t mca_base_alias_t;

OBJ_CLASS_DECLARATION(mca_base_alias_t);

OPAL_DECLSPEC int mca_base_alias_register (const char *project, const char *framework, const char *component_name, const char *component_alias, uint32_t alias_flags);

OPAL_DECLSPEC const mca_base_alias_t *mca_base_alias_lookup(const char *project, const char *framework, const char *component_name);

#endif /* OPAL_MCA_BASE_ALIAS_H */
