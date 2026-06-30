void launchSoftmaxV0(float* h_output, const float* h_input, const float* h_mask,
                     int batch_size, int num_heads, int seq_q, int seq_k,
                     float scale, int is_causal);

void launchSoftmaxV1(float* h_output, const float* h_input, const float* h_mask,
                     int batch_size, int num_heads, int seq_q, int seq_k,
                     float scale, int is_causal);

void launchSoftmaxV2(float* h_output, const float* h_input, const float* h_mask,
                     int batch_size, int num_heads, int seq_q, int seq_k,
                     float scale, int is_causal);