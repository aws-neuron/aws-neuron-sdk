/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <nrt/nrt_status.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEC_MAX_CHANNELS 32 /* matches MAXCHANNELS in NCCL */
#define NEC_MAX_CHANNEL_BUFFER_N 32 /* Channel buffers for reduce operation */
#define NEC_MAX_FOLD_N 8

#define NEC_CACHE_LINE_SIZE 128

/* Rank ID to denote network connector */
#define NEC_NET_CONNECTOR_RANK -1

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
 * On sender side, device increases host index to request data transfer to the remote device. Once
 * the network transaction is completed, host proxy will increase send credit on device via the
 * pointers in the structures.
 *
 * On receiver side, once data arrives from sender, host proxy will increase receive counter on
 * device to notify data is available on device memory. If the buffer is consumed by device, device
 * will increase host index to post next receive.
 */
struct enc_net_host_memory {
    union {
        struct {
            struct enc_net_host_memory_index post_recv[NEC_MAX_FOLD_N];
            volatile uint32_t *inc_recv_cnt;
            volatile uint32_t *dec_recv_cnt;
        } recv;
        struct {
            struct enc_net_host_memory_index post_send[NEC_MAX_FOLD_N];
            volatile uint32_t *inc_send_credit;
            volatile uint32_t *dec_send_credit;
        } send;
    };
};

/**
 * Network connector structure containing allocated resources for network transport
 */
struct enc_net_connector {
    int fold_n;

    /* Host memory */
    struct enc_net_host_memory *host_mem; /* host-accessible mapped pointer */
    dma_addr_t host_mem_addr; /* device-accessible physical address */
    void *host_mem_mhandle;

    /* Network transport buffer, allocated only for sender */
    int buf_n;
    uint32_t buf_sz;
    void *buf_mhandle;
    void *nccl_mhandle;
};

struct enc_ring {
    int prev;
    int next;
    int *user_ranks;

    /* Applicable only in case of remote neighbor */
    struct enc_net_connector *net_recv; /* if prev == NEC_NET_CONNECTOR_RANK */
    struct enc_net_connector *net_send; /* if next == NEC_NET_CONNECTOR_RANK */
};

struct enc_channel {
    /*
     * Application parameters for init
     */
    struct enc_ring ring;
    int id;

    /*
     * Neuron Runtime context
     */
    void *buf_mhandle;
    void *nccl_mhandle;

    struct enc_channel_context *ch_ctx;
};

struct enc_peer_info {
    int rank;
    int neuron_dev;
};

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
    /*
     * Application parameters for init
     */
    int rank;
    int rank_n;
    int neuron_dev;

    int node;
    int node_n;

    struct enc_peer_info *peers;

    int channel_n; /* Number of channels */
    int channel_buffer_n; /* Channel buffer depth, applies to all channels */
    struct enc_channel channels[NEC_MAX_CHANNELS];

    size_t chunk_size; /* Unit of transfer, applies to all channels */

    void* nccl_comm; /* Backward reference to NCCL comm */
    struct enc_op_context *ops_ctx; /* Backward reference to operation queue */
    struct encd_drv_comm *drv_comm; /* Reference to driver comm */
};

/**
 * Network transport FIFOs
 *
 * Host send proxy should know the size of each data chunk to send to remote device and recv proxy
 * needs destination addresses for each data from sender to submit network receive request.
 *
 * Such information is recorded when operation is loaded and becomes available on execution. Host
 * proxy uses these APIs to query the recorded FIFO.
 */
NRT_STATUS nec_get_net_size_fifo(nec_channel_t ch, void **base_addr, uint32_t **size_fifo, int *fifo_n);

typedef struct net_dest_addr {
    dma_addr_t dev_addr;
    void *host_addr;
    void *nccl_mhandle;
} net_dest_addr_t;

NRT_STATUS nec_get_net_dest_addr_fifo(nec_channel_t ch, net_dest_addr_t **dest_addr_fifo, int *fifo_n);

/**
 * Recv Counter / Send Credit
 *
 * Host proxy will increase send credit or recv counter in each step. At the end of an operation,
 * host proxy would need to decrease it to restore to default value.
 */
void nec_inc_net_send_credit(struct enc_net_connector *conn, uint32_t val);
void nec_dec_net_send_credit(struct enc_net_connector *conn, uint32_t val);
void nec_inc_net_recv_cnt(struct enc_net_connector *conn, uint32_t val);
void nec_inc_net_recv_cnt(struct enc_net_connector *conn, uint32_t val);

/**
 * Qeury device information
 */
int nec_get_device_count(void);
const char *nec_get_device_pci_bdf(int neuron_dev);

/**
 * Query if two devices are connected via P2P
 */
int nec_get_peer_device(int neuron_dev, int port);

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
#ifdef __cplusplus
}
#endif
