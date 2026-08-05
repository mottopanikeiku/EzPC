#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RINGLPN_EMP_SILENT_ABI_VERSION 1u
#define RINGLPN_EMP_SILENT_REVISION \
  "2fca139ff1974c039422af545bd4681e8d55acc1"
#define RINGLPN_EMP_SILENT_SID_BYTES 32u

/* Return zero on success. Callbacks must transfer exactly n bytes or fail. */
typedef int (*ringlpn_emp_io_send_fn)(void *context, const void *data,
                                      size_t n);
typedef int (*ringlpn_emp_io_recv_fn)(void *context, void *data, size_t n);
typedef int (*ringlpn_emp_io_flush_fn)(void *context);

typedef struct ringlpn_emp_io {
  void *context;
  ringlpn_emp_io_send_fn send;
  ringlpn_emp_io_recv_fn recv;
  ringlpn_emp_io_flush_fn flush;
} ringlpn_emp_io;

typedef enum ringlpn_emp_direction {
  RINGLPN_EMP_STRAIGHT = 0,
  RINGLPN_EMP_REVERSED = 1
} ringlpn_emp_direction;

typedef enum ringlpn_emp_parameter {
  RINGLPN_EMP_FERRET_B13 = 13,
  RINGLPN_EMP_FERRET_B11 = 11
} ringlpn_emp_parameter;

typedef struct ringlpn_emp_config {
  uint32_t abi_version;
  uint32_t local_party;       /* 0 or 1 */
  uint32_t sender_party;      /* 0 or 1 */
  uint32_t direction;         /* ringlpn_emp_direction */
  uint32_t parameter;         /* b13 straight, b11 reversed */
  uint32_t threads;
  uint64_t declared_count;
  uint64_t capacity_floor;
  uint8_t sid[RINGLPN_EMP_SILENT_SID_BYTES];
  ringlpn_emp_io io;
} ringlpn_emp_config;

typedef struct ringlpn_emp_counters {
  uint64_t declared_count;
  uint64_t consumed_count;
  uint64_t prepared_capacity;
  uint64_t setup_bytes_sent;
  uint64_t setup_bytes_received;
  uint64_t correlation_bytes_sent;
  uint64_t correlation_bytes_received;
  uint64_t adjustment_bytes_sent;
  uint64_t adjustment_bytes_received;
  uint64_t ciphertext_bytes_sent;
  uint64_t ciphertext_bytes_received;
  uint64_t flushes;
  uint32_t begun;
  uint32_t ended;
  uint32_t poisoned;
} ringlpn_emp_counters;

typedef struct ringlpn_emp_handle ringlpn_emp_handle;

/* All functions are exception-safe across the C boundary. */
uint32_t ringlpn_emp_silent_abi_version(void);
const char *ringlpn_emp_silent_revision(void);
ringlpn_emp_handle *ringlpn_emp_silent_create(const ringlpn_emp_config *config,
                                               char *error, size_t error_size);
int ringlpn_emp_silent_begin(ringlpn_emp_handle *handle, char *error,
                             size_t error_size);
/* width_bits is exactly 1, 62, or 128. Element encoding is respectively
 * one byte, native uint64_t (top two bits zero), or 16 little-endian bytes.
 * The wire ciphertext stream contains exactly 2*n*width_bits meaningful bits. */
int ringlpn_emp_silent_send(ringlpn_emp_handle *handle, uint32_t width_bits,
                            const void *messages0, const void *messages1,
                            uint64_t n, char *error, size_t error_size);
int ringlpn_emp_silent_recv(ringlpn_emp_handle *handle, uint32_t width_bits,
                            const uint8_t *choices, void *messages,
                            uint64_t n, char *error, size_t error_size);
int ringlpn_emp_silent_end(ringlpn_emp_handle *handle, char *error,
                           size_t error_size);
int ringlpn_emp_silent_get_counters(const ringlpn_emp_handle *handle,
                                    ringlpn_emp_counters *out, char *error,
                                    size_t error_size);
void ringlpn_emp_silent_destroy(ringlpn_emp_handle *handle);

#ifdef __cplusplus
}
#endif
