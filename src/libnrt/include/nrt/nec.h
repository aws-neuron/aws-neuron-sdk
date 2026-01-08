/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <time.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "nrt/nrt_status.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEC_MAX_CHANNELS 32 /* matches MAXCHANNELS in NCCL */
#define NEC_MAX_NR_CHANNEL_CHUNKS 32 /* Channel buffers for reduce operation */
#define NEC_MAX_FOLD_N 8

/*
 * We can set max communicator to anything here but ultimately we will be
 * limited by how much HW resources (such as TOP_SP semaphores or NX DRAM
 * space etc) get used up as number of communicators go up.
 */
#define NEC_MAX_COMM_N 12   /* Max supported replica-groups in NEFF */

#define NEC_MAX_NET_BUFFERS (2 * NEC_MAX_COMM_N) /* 2(hier & ring) x (# replica groups) */

#define NEC_CACHE_LINE_SIZE 128

/* Rank ID to denote network connector */
#define NEC_NET_CONNECTOR_RANK -1
/* MLA dev ID to denote network connector */
#define NEC_NET_MLA_DEV -1
/* MLA dev ID to denote POD connector */
#define NEC_POD_MLA_DEV -2
/* Rank ID to denote an unknown connector -> possibly not reachable */
#define NEC_UNKNOWN_RANK -3
/* MLA dev ID to denote an unknown connector -> possibly not reachable */
#define NEC_UNKNOWN_MLA_DEV -3

/* the number of hierarchical cc pipeline stage */
#define NEC_HIER_CC_PIPELINE_STAGE_N    (3)

/* the max number of outgoing requests in the recv/send proxy */
#define NCCL_NET_NEURON_MAX_REQUESTS 128

/**
 * The maximum number of concurrent cc execution. As NCCL needs this
 * information, define the size in the common header file.
 */
#define NEC_MAX_STREAM_N       4

/**
 * The different types of ofi communicators that are in the netResources
 * object that is used in the recv/send proxy
 */
typedef enum ofi_comm_type {
    NET_SEND_COMM,
    NET_RECV_COMM,
    NET_RECV_LISTEN_COMM,
    LOCAL_RECV_COMM,
    LOCAL_SEND_COMM
} ofi_comm_type_t;

/**
 * Neuron Elastic Collectives (NEC)
 *
 * This is the main component for Neuron Elastic Collectives in Neuron Runtime
 * (NRT). This is to provide collective operations to applications offloaded by
 * the device including collective comm init, receiving (post) operations,
 * building resources for the operation, triggering the operation and polling
 * its completion.
 *
 *     +-----------------------+
 *     |  Collectives App      |
 *     +-----------------------+
 *     |  Collectives Library  |
 *     +-----------------------+
 *     |       NEC / NRT       |
 *     +-----------------------+
 *     |        DEVICE         |
 *     +-----------------------+
 *
 * TODO: ENC will be renamed to NEC
 */

typedef enum nec_op_type {
    NEC_OP_ADD     = 0x0,
    NEC_OP_FMA     = 0x1,
    NEC_OP_MAX     = 0x2,
    NEC_OP_MIN     = 0x3,
    NEC_OP_INVALID = 0xF,
} nec_op_type_t;

typedef enum nec_data_type {
    NEC_DTYPE_INVALID  = 0x0,
    NEC_DTYPE_FP8_E3   = 0xD,
    NEC_DTYPE_FP8_E4   = 0xE,
    NEC_DTYPE_FP8_E5   = 0xF,
    NEC_DTYPE_FP16     = 0x7,
    NEC_DTYPE_BFLOAT16 = 0x6,
    NEC_DTYPE_FP32     = 0xA,
    NEC_DTYPE_UINT8    = 0x3,
    NEC_DTYPE_UINT16   = 0x5,
    NEC_DTYPE_UINT32   = 0x9,
    NEC_DTYPE_UINT64   = 0x1,
    NEC_DTYPE_INT8     = 0x2,
    NEC_DTYPE_INT16    = 0x4,
    NEC_DTYPE_INT32    = 0x8,
    NEC_DTYPE_INT64    = 0xC,
} nec_data_type_t;

/* Translated from what KaenaDriver returns */
typedef enum nec_pod_type {
    NEC_POD_TYPE_NONE,
    NEC_POD_TYPE_P2P,
    NEC_POD_TYPE_SWITCH,
    NEC_POD_TYPE_INVALID
} nec_pod_type_t;

typedef struct enc_comm* nec_comm_t;
typedef struct enc_channel* nec_channel_t;
typedef uint64_t dma_addr_t;

struct enc_net_host_memory_index {
    union {
        volatile uint32_t index;
        char pad[NEC_CACHE_LINE_SIZE]; /* Avoid false-sharing */
    };
};

/**
 * Host memory structure for network transport
 *
 * The proxy-thread progress function first waits for the device to be ready by
 * polling host index on fold 0 until it is (-1). Once (-1) was polled, the
 * proxy-thread resets the host index to 0 and notify the device that the
 * proxy-thread is ready by incrementing the handshake semaphore by 1.
 *
 * On the sender side, the device increase the host index to post a buffer to
 * send to a remote device. The proxy-thread send progress function polls the
 * host index and send posted buffers to the respective remote device. The
 * proxy-thread polls for send requests completions and notifies the device on
 * these completions by increasing the send_complete semaphore by the amount of
 * completed send requests. The device may in response to this notification
 * increase the host index further to post additional buffers to send. The
 * proxy-thread recognize the last entry in the FIFO by the fact it is
 * specially marked (See mark_fifo_end())
 *
 * On the receiver side, the device increase the host index to post receive
 * buffers to be filled with data from a remote device. The proxy-thread recv
 * progress function polls the host index and post the receive buffers to the
 * network plugin. The proxy-thread polls for receive completions and notifies
 * the device on these completions by increasing the recv_complete semaphore by
 * the amount of completed recv requests. The device use this notification to
 * know that data is available for processing on the device memory. The device
 * may also in response to this notification increase the host index further to
 * post additional buffers to post as receive buffers. The proxy-thread
 * recognize the last entry in the FIFO by the fact it is specially marked.
 *
 * For the ring algorithm:
 * The sender's handshake and send_complete semaphores
 * are the send-credit semaphore.
 * The receiver's handshake and recv_complete semaphores are the recv-cnt
 * semaphore.
 *
 * For the mesh algorithm:
 * The handshake semaphore is the local-handshake event semaphore for both
 * sender and receiver.
 * The recevier's recv_complete semaphore is the broadcast event semaphore.
 * The sender's send_complete semaphore is the sync event semaphore.
 */
struct enc_net_host_memory {
    union {
        struct {
            struct enc_net_host_memory_index post_recv[NEC_MAX_FOLD_N];
        } recv;
        struct {
            struct enc_net_host_memory_index post_send[NEC_MAX_FOLD_N];
        } send;
    };
};

typedef struct enc_host_mem {
    void *mem_handle;
    void *va;
    dma_addr_t pa;
    size_t size;
} enc_host_mem_t;


typedef struct enc_host_mem_shared {
    enc_host_mem_t mem;
    int refcnt;
} enc_host_mem_shared_t;

/**
 * Network connector structure containing allocated resources for network transport
 */
struct enc_net_connector {
    int fold_n;

    enc_host_mem_t net_host_mem; /* Used to signal proxy thread */
    enc_host_mem_shared_t *dynamic_input_host_mem; /* Used to pass info only available during execution */

    /* Network transport buffer, allocated only for sender */
    void *devmem_res;
    void *nccl_mhandle;

    /* Address and mhandle for event semaphores and pre-registered buffers */
    void *inc_recv_sem_nccl_mhandle;
    uint32_t *inc_recv_sem_values_buffer;
    void *inc_recv_sem_values_buffer_mhandle;

    /*
     * NCCL network connector data structure. When one proxy worker is used for
     * the same type (recv or send) network operation, connector information
     * should be included in each transaction.
     */
    void *nccl_connector;
};

typedef enum enc_pattern {
    ENC_PATTERN_RING,
    ENC_PATTERN_MESH,
    ENC_PATTERN_INVALID,
} enc_pattern_t;

typedef enum enc_net_connectivity {
    ENC_CONNECTIVITY_MESH,
    ENC_CONNECTIVITY_RDH,
    ENC_CONNECTIVITY_DEFAULT
} enc_net_connectivity_t;

struct enc_channel {
    /*
     * Application parameters for init
     */
    int id;
    enc_pattern_t pattern;

    /* Applicable only in case of remote neighbor */
    struct enc_net_connector *net_recv; /* if receving from rank over the network */
    struct enc_net_connector *net_send; /* if sending to rank over the network */

    /*
     * Neuron Runtime context
     */
    void *devmem_res;
    /* Gateway buffer is allocated only when hybrid ring is supported */
    void *devmem_gw_buf_res;
    void *nccl_mhandle;

    dma_addr_t gw_recv_buffer;
    dma_addr_t gw_send_buffer;

    struct enc_channel_context *ch_ctx;
    struct encd_dma_channel *drv_channel;
};

struct enc_peer_info {
    int neuron_dev;
    int rid;
    int tpb_index;
    int pod_node_id;
};

typedef enum enc_topology_mode {
    ENC_TOPO_NULL = 0,
    ENC_TOPO_4_DEVS_IN_ROW,
    ENC_TOPO_4_DEVS_IN_COLUMN,
} enc_topology_mode_t;

struct enc_comm_info {
    int neuron_dev;
    int rank;
    int rank_n;
    int local_rank_n;

    int node;
    int node_n;

    enc_topology_mode_t enc_topo_mode;

    /* Pod information received from NCCL */
    bool enable_pod;
    int pod;
    int pod_n;
    int pod_node;
    int pod_node_n;

    struct enc_peer_info *peers;
};

struct enc_ring {
    int prev;
    int next;
    int *user_ranks;

    /* used by one_rank_per_device rings only */
    bool duplicate;
};

/* Kangaring */
#define NEC_KANGARING_MAX_NUM_RANKS (128)
#define KANGARING_NUM_SENG_PER_DEV  (4)
#define KANGARING_NUM_TPB_PER_DEV   (8)
#define KANGARING_MAX_SECONDARIES   (3)

enum SEngine {
    S0 = 0,
    S1 = 1,
    S2 = 2,
    S3 = 3,
    SENGS_PER_DIE = 2,
    SENGS_PER_MLA = 4
};

struct enc_kangaring {
    int vnc;                                        // virtual neuron core size
    int logical_path[NEC_KANGARING_MAX_NUM_RANKS];  // the logical kangaring path: p0 s0 p1 s1 ...
    int prev;                                       // upstream
    int next;                                       // downstream
    int port;                                       // port to go to next

    /* In VNC 2 case, this is the only peer. For primary ranks, it refer to their secondary rank;
     * for secondary ranks, this refer to their primary rank.
     * In VNC 1 case, it refers specifically to the peer over rmtv with the same tpb index.
     */
    int peer_rmtv;
    /* In VNC 1 case, we have these 2 additional peers.
     * peer_over_rmtv2 refers to the peer over rmtv with a different tpb index.
     * peer_local refers to the local peer with a different tpb index
     */
    int peer_rmtv2;
    int peer_local;

    int next_peer_rmtv;                              // next's peer over rmtv

    bool is_primary;                                // is self rank on data path?
    bool is_next_pcie;                              // is next primary reached via pcie or d2d?
    bool duplicate;                                 // is this a duplicate channel?
    bool pattern2;                                  // is pattern 2?
};

typedef enum metaring_type {
    RING,
    KANGARING,
    SINGLE_CYCLE_RING,
    RDH,
    INVALID_METARING
} metaring_type_t;

struct enc_alg_metaring {
    int channel_n;
    struct enc_channel channels[NEC_MAX_CHANNELS];

    struct enc_ring ring_ranks[NEC_MAX_CHANNELS];
    struct enc_kangaring kangaring_ranks[NEC_MAX_CHANNELS];
    metaring_type_t type;

    /* Does the group contain only on rank per device? This variable is set to true when NCCL
     * returns device level H-cycles to runtime. In this case, we will parse that device H-cycle
     * and generate ring paths on runtime side. We do this because we need to enforce certain
     * pre-defined patterns in the paths so that we avoid dead locks between concurrent groups.
     */
    bool one_rank_per_device;
    /* Hybrid ring is supported when RG have 4 H-cycles of one_rank_per_device */
    bool is_hybrid_ring;
    bool tokens_exchanged;    /* reinitialzed tokens from old metaring config*/
    bool deadlock_free_rank_list;

    struct enc_comm *comm; /* Backward reference to ENC comm */
    struct encd_alg_metaring *drv_alg;

    /* For use by src/tgt pairs only */
    bool skip_send;
    bool skip_recv;
};

/*
 * The order of the events matter here, so while adding a new event make sure the event is added
 * to the right section of the list
 * 
 * ENC_COMMON_NUM_EVENT_TYPE:                           contains all common events between RDH-Mesh or A2A-mesh
 * ENC_MESH_NUM_EVENT_TYPE-ENC_COMMON_NUM_EVENT_TYPE:   contains events used by mesh
 * ENC_A2A_NUM_EVENT_TYPE-ENC_MESH_NUM_EVENT_TYPE:      contains events used by A2A only
 * ENC_RDH_NUM_EVENT_TYPE-ENC_A2A_NUM_EVENT_TYPE:       contains events used by RDH only
 *
 */
typedef enum enc_mesh_event_type {
    EVT_SYNC,
    EVT_GLOBAL_HNDSHK,
    EVT_LOCAL_HNDSHK,
    EVT_INTER_GRP_BRDCST,
    EVT_FUNCTION_BARRIER_FIRST_COLL,
    EVT_FUNCTION_BARRIER_LAST_COLL,
    EVT_REDUCE_LOCAL_HNDSHK,
    EVT_INTRA_GRP_BRDCST,
    ENC_COMMON_NUM_EVENT_TYPE,

    ENC_MESH_NUM_EVENT_START = ENC_COMMON_NUM_EVENT_TYPE,
    EVT_REDUCE_COPY = ENC_COMMON_NUM_EVENT_TYPE,
    EVT_REDUCE_WRITE,
    EVT_INTER_GRP_BRDCST_2,
    ENC_MESH_NUM_EVENT_TYPE,

    ENC_A2A_NUM_EVENT_START = ENC_MESH_NUM_EVENT_TYPE,
    EVT_LOCAL_HNDSHK_1 = ENC_MESH_NUM_EVENT_TYPE,
    EVT_LOCAL_HNDSHK_2,
    EVT_GLOBAL_HNDSHK_1,
    EVT_INTER_GRP_BRDCST_1,
    EVT_INTRA_GRP_BRDCST_1,
    EVT_2DEV_BRDCST,
    EVT_2DEV_HNDSHK,
    EVT_COPY_FROM_HOST,
    ENC_A2A_NUM_EVENT_TYPE,

    ENC_RDH_NUM_EVENT_START = ENC_A2A_NUM_EVENT_TYPE,
    EVT_RH_STEP_0 = ENC_A2A_NUM_EVENT_TYPE,
    EVT_RH_STEP_1,
    EVT_RH_STEP_2,
    EVT_RH_STEP_3,
    EVT_RH_STEP_4,
    EVT_RH_STEP_5,
    EVT_RH_STEP_6,
    EVT_RH_STEP_7,
    EVT_RH_STEP_8,
    EVT_RH_STEP_9,
    EVT_RDH_LOCAL_HANDSHAKE = EVT_RH_STEP_9,
    EVT_RDH_AXES_HANDSHAKE,
    EVT_RD_STEP_0,
    EVT_RD_STEP_1,
    EVT_RD_STEP_2,
    EVT_RD_STEP_3,
    EVT_RD_STEP_4,
    EVT_RD_STEP_5,
    EVT_RD_STEP_6,
    EVT_RDH_AXES_HANDSHAKE_2,
    EVT_1DEV_RDH_STEP_1,
    EVT_1DEV_RDH_STEP_2,
    EVT_1DEV_RD_STEP_1,
    EVT_1DEV_RD_STEP_2,
    EVT_1DEV_RH_STEP_1,
    EVT_2DEV_RD_STEP_0,
    EVT_2DEV_RD_STEP_1,
    EVT_2DEV_RD_STEP_2,
    EVT_2DEV_RD_STEP_3,
    EVT_2DEV_RD_STEP_4,
    EVT_RDH_LOCAL_PEER_HANDSHAKE,
    ENC_RDH_NUM_EVENT_TYPE    // We assume each event is used only once
                              // Enforced by encd_init_mesh_event()
} enc_mesh_event_type_t;

#define ENC_MESH_MAX_NUM_EVENTS 64

#define KiB     (1024)
#define MiB     (1024 * KiB)
#define GiB     (1024 * MiB)
#define ENC_MESH_CHANNEL_BUF_MAX_SIZE (8 * MiB)

struct enc_mesh_nbr_grp {
    int *ranks;
    int ranks_n;
};

struct enc_mesh_event {
    struct enc_mesh_nbr_grp src_neighbor_grp;
    struct enc_mesh_nbr_grp dst_neighbor_grp;
    bool valid;
    enc_mesh_event_type_t evt_type;
};

typedef enum enc_alg_mesh_type {
    ENC_ALG_FULL_MESH,
    ENC_ALG_GROUPED_MESH,
    ENC_ALG_MESH_TRN2,
    ENC_ALG_MESH_INVALID
} enc_alg_mesh_type_t;

/* TODO: In a separate commit we will change this to a cpp
 * file so we can have classes
 */
#define ENC_MAX_OP_TYPES     (13)
struct enc_alg_mesh_subtype {
    struct enc_mesh_event events[ENC_MESH_MAX_NUM_EVENTS];
    int num_events;
    struct encd_alg_mesh_subtype *drv_mesh;
    struct enc_alg_mesh *mesh; /* backward reference */
    size_t op_max_limit[ENC_MAX_OP_TYPES]; /* upper limit below which we will use mesh */
    size_t op_min_limit[ENC_MAX_OP_TYPES]; /* lower limit above which we will use mesh */
    size_t op_max_limit_sbuf[ENC_MAX_OP_TYPES]; /* upper limit below which we will use mesh for 2D tensors */
    size_t op_min_limit_sbuf[ENC_MAX_OP_TYPES]; /* lower limit above which we will use mesh for 2D Tensors */
    bool no_inplace_support;
    bool is_use_chnl_buffer; /* Whether channel bufer will be used or not */
    bool is_rdh;
    bool is_single_step_mesh;
    bool is_two_step_pod_mesh;
    uint32_t alltoall_iteration;
};

#define ENC_MAX_MESH_SUBTYPES         (20)
#define ENC_MESH_MAX_NUM_DEVICES      (128)

struct enc_alg_mesh {
    enc_alg_mesh_type_t mesh_type;

    union {
        struct {
            uint32_t devid_to_rankid[ENC_MESH_MAX_NUM_DEVICES];
            /* Whether it is a single or a multi chip mesh */
            bool is_multi_chip;
        } trn2;
        struct {
            int num_non_net_node_local_groups;
        } trn1;
        struct {
            bool root_rank;
            int num_intra_group_roots;
            int local_root_ids[ENC_MESH_MAX_NUM_DEVICES];
            int global_root_ids[ENC_MESH_MAX_NUM_DEVICES];
        } inf2;
    };
    int group_id;
    int num_groups;
    /* Mesh uses only a single channel */
    struct enc_channel channel;
    struct enc_alg_mesh_subtype mesh_subtype[ENC_MAX_MESH_SUBTYPES];

    /* Holds maximum amt of data a single group is allowed to deposit into
     * the channel buffer. The definition of a group varies by platform type.
     * On TRN1, TRN2 a group currently consists of all or some ranks from a
     * single chip but on INF2 it refers to a collection of chips. The concept
     * of a group exists to avoid traffic replication on the wire by combining
     * input data from multiple ranks within a group before sending it outside
     * of the group. Therefore at the destination side we only receive a single
     * chunk of data per group.
     */
    size_t max_chbuf_space_per_group;
    /* Valid only for TRN2. For TRN2 to prevent AXI deadlock we avoid on-chip
     * routing at the destination chip and deposit data in the HBM closest to
     * the entry port. So the rank owning that HBM receives data on behalf of
     * other ranks on that same chip. This is why we need to carve out dedicated
     * channel buf space for each of the other s-engines on the same chip.
     */
    size_t max_chbuf_space_per_seng;
    /* Valid only for single step mesh where we directly copy the entire input
     * buffer into another rank's channel buffer.
     */
    size_t max_chbuf_space_per_rank;

    /* Whether to use double buffer to skip global handshake */
    bool double_buffer;

    /* Whether to build RDH */
    bool build_rdh;
    void *rdh_devmem_res;  /*intra rdh channel buffer */
    bool use_rdh_2dev_proxy;

    bool tokens_exchanged;    /* reinitialzed tokens from old mesh config*/

    bool use_net;               /* Whether inter-node mesh with network proxy is used or not */

    /* Backward references to NCCL comm and general cluster info.
     * These might come from enc_comm or enc_alg_hier
     */
    struct enc_nccl_comm_node *nccl_comm_node; /* Reference to NCCL comm */
    struct enc_comm_info *ci; /* General cluster information */

    struct enc_comm *comm; /* Backward reference to ENC comm */
    struct encd_alg_mesh *drv_alg;

    /*
     * DMA mapped memory to host dedicated for A2Av metadata available only during
     * execution.
     */
    enc_host_mem_t alltoallv_host_input;
};

struct enc_alg_hier {
    struct {
        struct enc_nccl_comm_node *nccl_comm_node;
        struct enc_comm_info ci;

        struct enc_alg_metaring ring;
        struct enc_alg_metaring kangaring;
        struct enc_alg_mesh mesh;
    } intra;

    struct {
        struct enc_nccl_comm_node *nccl_comm_node;
        struct enc_comm_info ci;

        struct enc_alg_metaring ring;
        struct enc_alg_metaring rdh;
        struct enc_alg_mesh mesh;
    } inter;

    struct {
        struct {
            struct enc_nccl_comm_node *nccl_comm_node;
            struct enc_comm_info ci;

            struct enc_alg_metaring ring;
        } stage[NEC_HIER_CC_PIPELINE_STAGE_N];
    } pipeline;

    void* devmem_res; /* Hierarchical Reduce Scatter uses intermediate buffer */

    struct enc_comm *comm; /* Backward reference to ENC comm */
    struct encd_alg_hier *drv_alg;
};

/**
 * Comm info to query from NCCL
 */
typedef struct nccl_comm_info {
    /* General cluster information */
    uint64_t cluster_id; // randomly generated id used to identify unique clusters in log metrics
    time_t epoch; // the epoch of the initial barrier at the start of a collectives execution. used when generating core dumps so that all ranks agree on a datetime.

    int neuron_dev;
    int rank;
    int rank_n;
    int local_rank_n;

    int node;
    int node_n;

    bool enable_pod;
    int pod;
    int pod_n;
    int pod_node;
    int pod_node_n;

    struct enc_peer_info *peers; /* Needs to be allocated before calling ncclGetCommInfo() or NULL if peers info is not needed */

    /* Ring algorithm information */
    int channel_n;
    struct enc_ring rings[NEC_MAX_CHANNELS];

    /* Kangaring algorithm information */
    int kangaring_channel_n;
    int* kangaring_paths[NEC_MAX_CHANNELS];

    /* Hamiltonian cycles of MLAs, used to construct 1-rank-per-mla rings */
    int mla_cycle_n;
    int* mla_cycles[NEC_MAX_CHANNELS];
} nccl_comm_info_t;

typedef struct enc_nccl_comm_node {
    void *nccl_comm;
    char *key;
    size_t key_sz;
    /* Tracking the graph information in the nccl_comm. We can use
     * ncclGetCommInfo() but it's expensive. Instead, simply track the graph
     * information here. This flag can only changed from true to false. The
     * other way is not possible.
     */
    bool disable_graph;
    bool global_nccl_comm_node;
    int refcnt;
    uint32_t stream_id;

    uint32_t num_local_participants;
    uint32_t num_local_leaders;
    uint32_t my_local_leader;
    uint32_t *local_participants;
    uint32_t *local_leaders;
    struct bp_barrier *local_barrier;
    bool intra_pod_interface; /* When intra-pod interface is used, we can't skip exeuction barrier */
} enc_nccl_comm_node_t;

/* Neuron Device information. This data structure is used to send the device information from KeanaRuntime to
 * KaenaNCCL for nccl communicator building.
 */
#define ENC_PROXY_HISTOGRAM_OUTPUT_PATH_LENGTH_MAX (128)
typedef struct enc_proxy_histogram_config {
    bool enable;
    size_t bucket_usecs;
    size_t num_buckets;
    size_t per_neff_warmup;
    size_t warmup;
    char output_path[ENC_PROXY_HISTOGRAM_OUTPUT_PATH_LENGTH_MAX];
} enc_proxy_histogram_config_t;
 
typedef struct enc_neuron_device_info {
    int nec_dev_id;
    int mla_idx;
    int tpb_idx;
    int host_device_id;
    int routing_id;
    uint64_t pod_id;
    nec_pod_type_t pod_type;
    uint32_t pod_node_id;
    uint32_t virtual_server_id;
    enc_proxy_histogram_config_t histogram_config;
} enc_neuron_device_info_t;

/**
 * Collective communicator corresponding to ncclComm structure
 *
 * enc_comm is the Collective Comm that holds all the necessary information to
 * execute an collective operation. This should be pre-set before operations are
 * posted mainly because of the topology information built upon physical
 * connectivity. Collective operations are executed on multiple channels and a
 * channel is a path for data transfer along a pre-built topology.
 */
struct enc_comm {
    struct enc_nccl_comm_node *nccl_comm_node; /* Reference to NCCL comm */
    struct enc_comm_info ci; /* General cluster information */
    int id;
    int stream_id;

    /*
     * Algorithms
     */
    struct enc_alg_metaring ring;
    struct enc_alg_metaring kangaring;
    struct enc_alg_metaring rdh;
    struct enc_alg_hier hier;
    struct enc_alg_mesh mesh;

    /**
     * Use these handles to share network connector buffers across NEFFs.
     * Only used in global comm. Other comms will refer to the global comm to reuse them.
     * We use net_conn_count to sequentially assign these reservations to network conectors
     * to make sure:
     * 1) different comm in a NEFF don't reuse the same buffer (for multi-stream cases)
     * 2) for each NEFF, we always start with index 0 and go up for the most overlap and
     *    reusability. We reset net_conn_count to 0 in enc_load_operations
     */
    int net_conn_count;
    void* net_connector_devmem_res[NEC_MAX_NET_BUFFERS];

    // TODO: nr_channel_chunks and chunk_size should not be a comm property anymore
    int nr_channel_chunks; /* Channel buffer depth, applies to all channels */
    size_t chunk_size; /* Unit of transfer, applies to all channels */

    struct encd_comm *drv_comm; /* Reference to driver comm */

    char topology[1024]; /* Used for debugging purposes only to print the topology in case of an error */
};

/**
 * Global communicator
 */
struct enc_glb_comm {
    uint32_t g_device_id; /* Same as comm->rank */
    uint32_t g_device_cnt; /* Same as comm->rank_n */
    uint32_t vtpb_idx;
    int nec_dev_id;
    int mla_idx;
    /* Absolute neuron device hw id. This is the ID that driver
       exposes neuron device on to host system aka OS. Neuron devices
       are expesed to RT by different ID in case docker remaps
       devices */
    int host_device_id;
    int routing_id;
    uint32_t virtual_server_id;
    nec_pod_type_t pod_type;
    uint32_t pod_node_id;
    uint32_t pod_sz;
    uint64_t pod_id;
    const char *root_comm_id; /* By getenv in nrt_config */
    bool check_sigs;          /* By getenv in nrt_config */

    uint32_t *rank_nodes; /* The node index of each rank */
    uint32_t *local_ranks; /* The intra-node rank of each rank */

    enc_nccl_comm_node_t nccl_comm_node; /* nccl_comm node can be used by any stream */

    struct bananaphone *local_rings;
    struct bp_handle *local_peer_handles;

    /**
     * A set of buffers containing values that are used to
     * increment semaphores over efa transactions.
     */
    uint32_t *inc_recv_sem_values_buffer;
    size_t inc_recv_sem_values_buffer_size;

    struct enc_comm comm;

    /* TODO: manage all the devmem reservations in a single place
     * Today we share the buffers under the below path:
     * enc_glb_comm->comm->ring.channels[ring_channel_id].devmem_res
     * We need to move the above reservations and the one below to a
     * singleton class e.g. enc_glb_comm->devmem_res_pool
     */
    void* inter_rdh_devmem_res[NEC_MAX_STREAM_N];
    /* TODO: manage all the devmem reservations in a single place
     * this mem res is referred by comm->rdh.rdh_devmem_res
     */
    void* intra_rdh_devmem_res[NEC_MAX_STREAM_N];

    void *gateway_devmem_res[NEC_MAX_STREAM_N][NEC_MAX_CHANNELS];

    pthread_mutex_t gcomm_setup_mtx;

    void *proxy_queue;   // opaque pointer to enc_proxy_queue

    void *device_barrier_table;
};

/**
 * Network transport FIFOs
 *
 * Host send proxy should know the EFA buffer index, offset in the buffer and the size of
 * each data tranfer to send to remote device and recv proxy
 * needs destination addresses for each data from sender to submit network receive request.
 * Send and recv proxy should know when to report the completion of using
 * EFA buffer and complete is used to notify it.
 *
 * Such information is recorded when operation is loaded and becomes available on execution. Host
 * proxy uses these APIs to query the recorded FIFO.
 */

/**
 * A net_ops_info_t entry corresponds to a set of smaller operations that are defined by multiple
 * net_src_addr_t and net_dest_addr_t. These sub operations can correspond to different types of
 * actions, so store a net_addr_mark_t identifier in each net_src_addr_t or net_dest_addr_t entry
 * to denote the purpose of the sub-operation.
 */
typedef enum net_addr_mark {
    NET_TRANSFER,       /* Will drive data transfer over EFA */
    NET_OP_COMPLETE,    /* Will mark final completion of a collective operation */
    EXEC_COMPLETE       /* Will mark final completion of a collective load execution */
} net_addr_mark_t;

typedef struct net_src_addr {
    uint32_t net_op_idx;
    int complete;
    dma_addr_t dev_addr;
    void *host_addr;
    void *nccl_mhandle;
    uint32_t size;
    net_addr_mark_t mark;
    void* proxy_histogram_tag;
     /* Fields below are for mesh only */
    int dst_rank;
} net_src_addr_t;

typedef struct net_dest_addr {
    uint32_t net_op_idx;
    int complete;
    dma_addr_t dev_addr;
    void *host_addr;
    void *nccl_mhandle;
    uint32_t size;
    net_addr_mark_t mark;
    /* Fields below are for mesh only */
    int src_rank;
} net_dest_addr_t;

typedef struct net_ops_info {
    uint16_t sema_shift_offset;
    bool early_send_completion;
    bool early_recv_posting;
    volatile uint32_t *inc_send_handshake;
    volatile uint32_t *inc_send_complete;
    volatile uint32_t *inc_recv_handshake;
    volatile uint32_t *inc_recv_complete;
    uint32_t tx_entry_cnt;
    uint32_t rx_entry_cnt;
    uint32_t net_idx_loop_size;
    uint32_t initial_send_credits;
    uint32_t ending_recv_credits;
    size_t data_type_sz;
    bool is_dynamic_send_recv_sz;
    bool is_recv_sz_known_by_dst;
    bool variable_peer;
    bool add_to_histogram;
    /*
     * proxy uses this pointer to get connector information from transaction
     * saddr/daddr fifo entry of each operation.
     */
    void *enc_channel;
} net_ops_info_t;

/**
 * API for proxy-thread to increase handshake and send/recv semaphores by writing directly to the
 * memory mapped semaphore inc register.
 * For more information, see documentation on struct enc_net_host_memory definition.
 */
void nec_inc_semaphore(volatile uint32_t *sem_inc_addr, uint32_t val);

/**
 * API for proxy-thread to get dynamic send and offset for the case where message
 * size is determined by data only available during execution.
 */
size_t nec_get_dynamic_send_size_bytes(enc_host_mem_t *dyn_input, size_t data_type_sz, int dst_rank, int rank_n);
size_t nec_get_dynamic_send_offset_bytes(enc_host_mem_t *dyn_input, size_t data_type_sz, int dst_rank, int rank_n);
size_t nec_get_dynamic_recv_size_bytes(enc_host_mem_t *dyn_input, size_t data_type_sz, int src_rank, int rank_n);
size_t nec_get_dynamic_recv_offset_bytes(enc_host_mem_t *dyn_input, size_t data_type_sz, int src_rank, int rank_n);
void nec_set_recv_size_bytes(enc_host_mem_t *dyn_input, size_t recv_size_bytes, size_t data_type_sz, int src_rank, int rank_n);

/**
 * Qeury device information
 */
int nec_get_device_count(int *available_devices_array, uint32_t array_size);
int nec_get_device_pci_bdf(int neuron_dev, uint32_t *domain, uint32_t *bus_num, uint8_t *pci_slot, uint8_t *dev_func);

/**
 * Query vcore size
 */
NRT_STATUS nec_get_virtual_core_size(uint32_t *virtual_core_size);

typedef struct nec_version_info {
    uint64_t major;
    uint64_t minor;
    uint64_t patch;
    uint64_t maintenance;
    char git_hash[16];
    uint64_t compatibility_version;
    // Any new fields added needs to be here. The fields before this cannot be
    // changed to maintain backward compatibility
    uint8_t future_fields[];
} nec_version_info_t;

NRT_STATUS nec_get_version_info(nec_version_info_t *version_info);

NRT_STATUS nec_build_port_and_rid_map(int local_nec_dev_id, int *mla_indexes, int *host_device_ids, int count);

bool nec_is_mla_available(int local_nec_dev_id, int mla_idx);

int nec_mla_idx_to_rid(int local_nec_dev_id, int mla_idx);

int nec_rid_to_mla_idx(int local_nec_dev_id, int rid);

int nec_get_peer_mla_idx(int local_nec_dev_id, int mla_idx, int port);

int nec_get_p2p_pod_peer_node(uint32_t nec_dev_id, int node, uint32_t port_distance,
                              int *peer_node);

NRT_STATUS nec_pod_node_can_access_peer_node(nec_pod_type_t pod_type,
                                             uint32_t local_rid, uint32_t local_node_id,
                                             uint32_t remote_rid, uint32_t remote_node_id,
                                             int *can_access_peer);

void nec_ndl_printk(char *str, uint32_t size, uint32_t action);

#ifdef __cplusplus
}
#endif
