/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2014-2018 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2018      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2018      Amazon.com, Inc. or its affiliates.  All Rights reserved.
 * Copyright (c) 2018-2024 Triad National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2019-2025 Google, LLC. All rights reserved.
 * Copyright (c) 2019      Intel, Inc.  All rights reserved.
 * Copyright (c) 2022      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "btl_uct.h"

static int mca_btl_uct_module_register_mca_params(mca_btl_uct_module_t *module)
{
    mca_base_component_t dummy_component = mca_btl_uct_component.super.btl_version;
    snprintf(dummy_component.mca_component_name, sizeof(dummy_component.mca_component_name), "uct_%s", module->md->desc.md_name);

    char *tmp = module->all_transports;
    (void) mca_base_component_var_register(
        &dummy_component, "available_transports",
        "Comma-delimited list of available transports in this module",
        MCA_BASE_VAR_TYPE_STRING, NULL, /*bind=*/0, /*flags=*/0,
        OPAL_INFO_LVL_3, MCA_BASE_VAR_SCOPE_CONSTANT, &module->all_transports);
    free(tmp);
    
    module->allowed_transports = mca_btl_uct_component.allowed_transports;
    (void) mca_base_component_var_register(
        &dummy_component, "transports",
        "Comma-delimited list of transports to use sorted by increasing "
        "priority. The list of transports available can be queried using ucx_info. Special"
        "values: any (any available) (default: dc_mlx5,rc_mlx5,ud,any)",
        MCA_BASE_VAR_TYPE_STRING, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE, OPAL_INFO_LVL_3,
        MCA_BASE_VAR_SCOPE_LOCAL, &module->allowed_transports);

    return mca_btl_base_param_register(&dummy_component, &module->super);
}

static int mca_btl_uct_set_all_transports(mca_btl_uct_module_t *module, uct_tl_resource_desc_t *tl_desc, unsigned num_tls)
{
    size_t all_transports_size = 0;
    for (int i = 0 ; i < num_tls ; ++num_tls) {
        all_transports_size += strlen(tl_desc[i].tl_name) + 1;
    }
    module->all_transports = calloc(1, all_transports_size);
    if (NULL == module->all_transports) {
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    for (int i = 0 ; i < num_tls ; ++num_tls) {
        strncat(module->all_transports, tl_desc[i].tl_name, all_transports_size);
        all_transports_size -= strlen(tl_desc[i].tl_name);
        if (i < num_tls - 1) {
            strncat(module->all_transports, ",", all_transports_size);
            all_transports_size -= 1;
        }
    }

    return OPAL_SUCCESS;
}

static int mca_btl_uct_create_tl(mca_btl_uct_md_t *md, uct_tl_resource_desc_t *tl_desc)
{
    if (tl_desc->.dev_type != UCT_DEVICE_TYPE_NET) {
	return OPAL_ERR_NOT_SUPPORTED;
    }

    mca_btl_uct_tl_t *tl = OBJ_NEW(mca_btl_uct_tl_t);
    if (OPAL_UNLIKELY(NULL == tl)) {
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    /* initialize btl tl structure */
    tl->uct_md = md;
    OBJ_RETAIN(md);
    tl->uct_tl_name = strdup(tl_desc->tl_name);
    tl->uct_dev_name = strdup(tl_desc->dev_name);

    ucs_status_t ucs_status = uct_md_iface_config_read(md->uct_md, tl_desc->tl_name, /*env_prefix=*/NULL,
						       /*filename=*/NULL, &tl->uct_tl_config);
    if (UCS_OK != ucs_status) {
      OBJ_RELEASE(tl);
      return OPAL_ERROR;
    }

    uct_worker_h uct_worker;
    ucs_status = uct_worker_create(module->ucs_async, UCS_THREAD_MODE_SINGLE, &uct_worker);
    if (OPAL_UNLIKELY(UCS_OK != ucs_status)) {
      OBJ_RELEASE(tl);
      return OPAL_ERROR;
    }

    tl->iface_params.open_mode = UCT_IFACE_OPEN_MODE_DEVICE;
    tl->iface_params.mode = {
        .device = {
	    .tl_name = tl->uct_tl_name,
	    .dev_name = tl->uct_dev_name,
        }
    };

#if UCT_API >= UCT_VERSION(1, 6)
    tl->iface_params.field_mask = UCT_IFACE_PARAM_FIELD_OPEN_MODE | UCT_IFACE_PARAM_FIELD_DEVICE;
#endif
    uct_iface_t uct_iface;
    ucs_status = uct_iface_open(md->uct_md, uct_worker, &tl->iface_params,
                                tl->uct_tl_config, &uct_iface);
    if (OPAL_UNLIKELY(UCS_OK != ucs_status)) {
      uct_worker_destroy(uct_worker);
      OBJ_RELEASE(tl);
      return OPAL_ERROR;
    }

    /* only need to query one of the interfaces to get the attributes */
    ucs_status = uct_iface_query(context->uct_iface, &tl->uct_iface_attr);

    /* No longer need these for now so go ahead and destroy them */
    uct_iface_close(uct_iface);
    uct_worker_destroy(uct_worker);

    if (UCS_OK != ucs_status) {
      OBJ_RELEASE(tl);
      return OPAL_ERROR;
    }

    opal_list_append(&md->tls, tl);
    
    BTL_VERBOSE(("Interface CAPS for tl %s::%s: 0x%lx", md->desc->md_name, tl_desc->tl_name,
                 (unsigned long) MCA_BTL_UCT_TL_ATTR(tl, 0).cap.flags));

    return OPAL_SUCCESS;
}

static int mca_btl_uct_query_tls(mca_btl_uct_module_t *module, mca_btl_uct_md_t *md,
				 uct_tl_resource_desc_t *tl_descs, unsigned tl_count)
{
    for (unsigned i = 0; i < tl_count; ++i) {
        BTL_VERBOSE(("processing tl %s -> %s", md->desc->md_name, tl_descs[i].tl_name));
        if (NULL != module) {
        }

	int rc = mca_btl_uct_create_tl(md, tl_descs + i);
	if (OPAL_SUCCESS != rc && OPAL_ERR_NOT_SUPPORTED != rc) {
	    return rc;
	}
    }

    return OPAL_SUCCESS;
}

#if UCT_API >= UCT_VERSION(1, 7)
static int mca_btl_uct_component_process_uct_md(uct_component_h component,
                                                uct_md_resource_desc_t *md_desc)
#else
static int mca_btl_uct_component_process_uct_md(uct_md_resource_desc_t *md_desc)
#endif
{
    mca_rcache_base_resources_t rcache_resources;
    uct_tl_resource_desc_t *tl_desc;
    mca_btl_uct_module_t *module = NULL;
    uct_md_config_t *uct_config;
    uct_md_attr_t md_attr;
    mca_btl_uct_md_t *md;
    int list_rank;
    unsigned num_tls;
    char *tmp;
    ucs_status_t ucs_status;
    int connection_list_rank = -1;
    bool consider_for_connection_module = false;

    BTL_VERBOSE(("processing memory domain %s", md_desc->md_name));

    if (MCA_BTL_UCT_MAX_MODULES == mca_btl_uct_component.module_count) {
        BTL_VERBOSE(("created the maximum number of allowable modules"));
        return OPAL_ERR_NOT_AVAILABLE;
    }

    BTL_VERBOSE(("checking if %s should be used for communication", md_desc->md_name));
    list_rank = mca_btl_uct_include_list_rank (md_desc->md_name, &mca_btl_uct_component.memory_domain_list);

    if (list_rank < 0) {
        BTL_VERBOSE(("checking if %s should be used for connections", md_desc->md_name));
        connection_list_rank = mca_btl_uct_include_list_rank (md_desc->md_name, &mca_btl_uct_component.connection_domain_list);

        if (connection_list_rank < 0) {
            /* nothing to do */
            BTL_VERBOSE(("not continuing with memory domain %s", md_desc->md_name));
            return OPAL_SUCCESS;
        }

        BTL_VERBOSE(("will be considering domain %s for connections only", md_desc->md_name));
        consider_for_connection_module = true;
    }

    md = OBJ_NEW(mca_btl_uct_md_t);

#if UCT_API >= UCT_VERSION(1, 7)
    ucs_status = uct_md_config_read(component, NULL, NULL, &uct_config);
    if (UCS_OK != ucs_status) {
        BTL_VERBOSE(("uct_md_config_read failed %d (%s)", ucs_status, ucs_status_string(ucs_status)));
        return OPAL_ERR_NOT_AVAILABLE;
    }
    ucs_status = uct_md_open(component, md_desc->md_name, uct_config, &md->uct_md);
    if (UCS_OK != ucs_status) {
        BTL_VERBOSE(("uct_md_open failed %d (%s)", ucs_status, ucs_status_string(ucs_status)));
        return OPAL_ERR_NOT_AVAILABLE;
    }
#else
    ucs_status = uct_md_config_read(md_desc->md_name, NULL, NULL, &uct_config);
    if (UCS_OK != ucs_status) {
        BTL_VERBOSE(("uct_md_config_read failed %d (%s)", ucs_status, ucs_status_string(ucs_status)));
        return OPAL_ERR_NOT_AVAILABLE;
    }
    ucs_status = uct_md_open(md_desc->md_name, uct_config, &md->uct_md);
    if (UCS_OK != ucs_status) {
        BTL_VERBOSE(("uct_md_open failed %d (%s)", ucs_status, ucs_status_string(ucs_status)));
        return OPAL_ERR_NOT_AVAILABLE;
    }
#endif
    uct_config_release(uct_config);

    ucs_status = uct_md_query(md->uct_md, &md_attr);
    if (UCS_OK != ucs_status) {
        BTL_VERBOSE(("uct_config_release failed %d (%s)", ucs_status, ucs_status_string(ucs_status)));
        return OPAL_ERR_NOT_AVAILABLE;
    }
    ucs_status = uct_md_query_tl_resources(md->uct_md, &tl_desc, &num_tls);
    if (UCS_OK != ucs_status) {
        BTL_VERBOSE(("uct_config_release failed %d (%s)", ucs_status, ucs_status_string(ucs_status)));
        return OPAL_ERR_NOT_AVAILABLE;
    }

    if (!consider_for_connection_module) {
        module = mca_btl_uct_alloc_module(md_desc->md_name, md, md_attr.rkey_packed_size);
        if (NULL == module) {
            uct_release_tl_resource_list(tl_desc);
            return OPAL_ERR_OUT_OF_RESOURCE;
        }

        rc = btl_uct_set_all_transports(module, tl_desc, num_tls);
        if (OPAL_SUCCESS != rc) {
            uct_release_tl_resource_list(tl_desc);
            mca_btl_uct_finalize(&module->super);
            return OPAL_ERR_OUT_OF_RESOURCE;
        }

        btl_uct_module_register_mca_params(module);
    }

    /* if this module is not to be used for communication check if it has a transport suitable 
     * for forming connections. */
    (void) mca_btl_uct_query_tls(module, md, tl_desc, num_tls, consider_for_connection_module);
    uct_release_tl_resource_list(tl_desc);

    if (opal_list_get_size(&md->tls)) {
        /* no suitable tls */
        OBJ_RELEASE(md);
        if (NULL != module) {
            mca_btl_uct_finalize(&module->super);
        }
        return OPAL_ERR_NOT_AVAILABLE;
    }

    /* release the initial reference to the md object. if any modules were created the UCT md will
     * remain open until those modules are finalized. */
    OBJ_RELEASE(md);

    return OPAL_SUCCESS;
}

#if UCT_API >= UCT_VERSION(1, 7)
static int mca_btl_uct_component_process_uct_component(uct_component_h component)
{
    uct_component_attr_t attr = {.field_mask = UCT_COMPONENT_ATTR_FIELD_NAME
                                               | UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT};
    ucs_status_t ucs_status;
    int rc;

    ucs_status = uct_component_query(component, &attr);
    if (UCS_OK != ucs_status) {
        return OPAL_ERROR;
    }

    BTL_VERBOSE(("processing uct component %s", attr.name));

    attr.md_resources = calloc(attr.md_resource_count, sizeof(*attr.md_resources));
    attr.field_mask |= UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    ucs_status = uct_component_query(component, &attr);
    if (UCS_OK != ucs_status) {
        return OPAL_ERROR;
    }

    for (unsigned i = 0; i < attr.md_resource_count; ++i) {
        rc = mca_btl_uct_component_process_uct_md(component, attr.md_resources + i);
        if (OPAL_SUCCESS != rc) {
            break;
        }
    }

    free(attr.md_resources);

    return OPAL_SUCCESS;
}
#endif /* UCT_API >= UCT_VERSION(1, 7) */

static void mca_btl_uct_component_validate_modules(void) {
    if (mca_btl_uct_component.conn_module != NULL) {
        /* verify that a connection-only module is required. this might be the case in some systems
         * where rc verbs is avaiable but ud is not. */
        bool need_conn_module = false;
        for (int i = 0 ; i < mca_btl_uct_component.module_count ; ++i) {
            mca_btl_uct_module_t *module =  mca_btl_uct_component.modules[i];
            if (module->conn_tl != NULL) {
                continue;
            }
            if ((module->rdma_tl && mca_btl_uct_tl_requires_connection_tl(module->rdma_tl)) ||
                (module->am_tl && mca_btl_uct_tl_requires_connection_tl(module->am_tl))) {
                need_conn_module = true;
                break;
            }
        }

        if (!need_conn_module) {
            mca_btl_uct_finalize (&mca_btl_uct_component.conn_module->super);
            mca_btl_uct_component.conn_module = NULL;
        }
    } else {
        int usable_module_count = mca_btl_uct_component.module_count;

        /* check that all modules can be used */
        for (int i = 0 ; i < mca_btl_uct_component.module_count ; ++i) {
            mca_btl_uct_module_t *module =  mca_btl_uct_component.modules[i];
            if (NULL != module->conn_tl) {
                /* module has its own connection transport */
                continue;
            }

            if (((module->rdma_tl && mca_btl_uct_tl_requires_connection_tl(module->rdma_tl)) ||
                 (module->am_tl && mca_btl_uct_tl_requires_connection_tl(module->am_tl)))
                && NULL == module->conn_tl) {
                /* module can not be used */
                BTL_VERBOSE(("module for memory domain %s can not be used due to missing connection transport",
                             module->md_name));
                mca_btl_uct_finalize (&mca_btl_uct_component.modules[i]->super);
                mca_btl_uct_component.modules[i] = NULL;
            }
        }

        /* remove holes in the module array */
        if (usable_module_count < mca_btl_uct_component.module_count) {
            for (int i = 0 ; i < mca_btl_uct_component.module_count ; ++i) {
                if (mca_btl_uct_component.modules[i] == NULL) {
                    for (int j = i ; j < mca_btl_uct_component.module_count ; ++j) {
                        mca_btl_uct_component.modules[i++] = mca_btl_uct_component.modules[j];
                    }
                }
            }
            mca_btl_uct_component.module_count = usable_module_count;
        }
    }
}

/*
 *  UCT component initialization:
 *  (1) read interface list from kernel and compare against component parameters
 *      then create a BTL instance for selected interfaces
 *  (2) setup UCT listen socket for incoming connection attempts
 *  (3) register BTL parameters with the MCA
 */

static mca_btl_base_module_t **mca_btl_uct_component_init(int *num_btl_modules,
                                                          bool enable_progress_threads,
                                                          bool enable_mpi_threads)
{
    /* for this BTL to be useful the interface needs to support RDMA and certain atomic operations
     */
    struct mca_btl_base_module_t **base_modules;
    ucs_status_t ucs_status;
    int rc;

    BTL_VERBOSE(("initializing uct btl"));

    if (NULL == mca_btl_uct_component.memory_domains
        || 0 == strlen(mca_btl_uct_component.memory_domains)
        || 0 == strcmp(mca_btl_uct_component.memory_domains, "none")) {
        BTL_VERBOSE(("no uct memory domains specified"));
        return NULL;
    }

    mca_btl_uct_component_parse_include_list(mca_btl_uct_component.memory_domains,
                                             &mca_btl_uct_component.memory_domain_list);
    mca_btl_uct_component_parse_include_list(mca_btl_uct_component.allowed_transports,
                                             &mca_btl_uct_component.allowed_transport_list);
    mca_btl_uct_component_parse_include_list(mca_btl_uct_component.connection_domains,
                                             &mca_btl_uct_component.connection_domain_list);

    mca_btl_uct_component.module_count = 0;

#if UCT_API >= UCT_VERSION(1, 7)
    uct_component_h *components;
    unsigned num_components;

    ucs_status = uct_query_components(&components, &num_components);
    if (UCS_OK != ucs_status) {
        BTL_ERROR(("could not query UCT components"));
        return NULL;
    }

    /* generate all suitable btl modules */
    for (unsigned i = 0; i < num_components; ++i) {
        rc = mca_btl_uct_component_process_uct_component(components[i]);
        if (OPAL_SUCCESS != rc) {
            break;
        }
    }

    uct_release_component_list(components);

#else /* UCT 1.6 and older */
    uct_md_resource_desc_t *resources;
    unsigned resource_count;

    uct_query_md_resources(&resources, &resource_count);

    /* generate all suitable btl modules */
    for (unsigned i = 0; i < resource_count; ++i) {
        rc = mca_btl_uct_component_process_uct_md(resources + i);
        if (OPAL_SUCCESS != rc) {
            break;
        }
    }

    uct_release_md_resource_list(resources);

#endif /* UCT_API >= UCT_VERSION(1, 7) */

    /* filter out unusable modules before sending the modex */
    mca_btl_uct_component_validate_modules();

    mca_btl_uct_modex_send();

    /* pass module array back to caller */
    base_modules = calloc(mca_btl_uct_component.module_count, sizeof(*base_modules));
    if (NULL == base_modules) {
        return NULL;
    }

    memcpy(base_modules, mca_btl_uct_component.modules,
           mca_btl_uct_component.module_count * sizeof(mca_btl_uct_component.modules[0]));

    *num_btl_modules = mca_btl_uct_component.module_count;

    BTL_VERBOSE(("uct btl initialization complete. found %d suitable memory domains",
                 mca_btl_uct_component.module_count));

    return base_modules;
}
