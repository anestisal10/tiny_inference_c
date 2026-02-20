#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- Hyperparameters (Must match Python Script) ---
#define VOCAB       50257
#define DIM         1024
#define N_LAYERS    24
#define N_HEADS     16
#define MAX_POS     1024
#define EPS         1e-5f
#define SEQ_LEN     64    // Max tokens to generate
#define MAX_TOKEN_LEN 256  // Max characters per word in vocab
#define DK          (DIM / N_HEADS)
#define DFF         (DIM * 4)
#define MASK_VAL    -1e9f

// Global Weights Pointers
float *w_emb_word, *w_emb_pos, *ln_f_g, *ln_f_b;
float *w_q_all, *b_q_all, *w_k_all, *b_k_all, *w_v_all, *b_v_all;
float *w_o_all, *b_o_all, *ln_1_g_all, *ln_1_b_all;
float *w_ff1_all, *b_ff1_all, *w_ff2_all, *b_ff2_all, *ln_2_g_all, *ln_2_b_all;

// Global Vocab Array
char vocabulary[VOCAB][MAX_TOKEN_LEN];

// --- Utilities ---
void matmul(const float* A, const float* B, const float* bias, float* C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = (bias) ? bias[n] : 0.0f;
            for (int k = 0; k < K; k++) sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
    }
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

void layer_norm(const float* in, const float* gamma, const float* beta, float* out, int seq_n) {
    for (int s = 0; s < seq_n; s++) {
        const float* vec = in + s * DIM;
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < DIM; i++) mean += vec[i];
        mean /= DIM;
        for (int i = 0; i < DIM; i++) { float d = vec[i] - mean; var += d*d; }
        var /= DIM;
        float inv_std = 1.0f / sqrtf(var + EPS);
        for (int i = 0; i < DIM; i++) out[s * DIM + i] = (vec[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

void softmax(float* x, int N) {
    float max_val = x[0];
    for (int n = 1; n < N; n++) if (x[n] > max_val) max_val = x[n];
    float sum = 0.0f;
    for (int n = 0; n < N; n++) { x[n] = expf(x[n] - max_val); sum += x[n]; }
    for (int n = 0; n < N; n++) x[n] /= sum;
}

// --- Inference Engine ---
// Returns the ID of the next predicted token
int predict_next_token(int* input_ids, int seq_len) {
    // 1. Alloc Temp Buffers
    float* x      = (float*)calloc(seq_len * DIM, sizeof(float));
    float* x_norm = (float*)calloc(seq_len * DIM, sizeof(float));
    float* attn   = (float*)calloc(N_HEADS * seq_len * seq_len, sizeof(float));
    float* q      = (float*)calloc(seq_len * DIM, sizeof(float));
    float* k      = (float*)calloc(seq_len * DIM, sizeof(float));
    float* v      = (float*)calloc(seq_len * DIM, sizeof(float));
    float* logits = (float*)calloc(VOCAB, sizeof(float));

    // 2. Embeddings
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < DIM; d++) {
            x[s*DIM + d] = w_emb_word[input_ids[s]*DIM + d] + w_emb_pos[s*DIM + d];
        }
    }

    // 3. Layers
    for (int l = 0; l < N_LAYERS; l++) {
        // --- Attention ---
        layer_norm(x, ln_1_g_all + l*DIM, ln_1_b_all + l*DIM, x_norm, seq_len);
        
        matmul(x_norm, w_q_all + l*DIM*DIM, b_q_all + l*DIM, q, seq_len, DIM, DIM);
        matmul(x_norm, w_k_all + l*DIM*DIM, b_k_all + l*DIM, k, seq_len, DIM, DIM);
        matmul(x_norm, w_v_all + l*DIM*DIM, b_v_all + l*DIM, v, seq_len, DIM, DIM);

        float scale = 1.0f / sqrtf(DK);
        for (int h=0; h<N_HEADS; h++) {
            for (int i=0; i<seq_len; i++) {
                for (int j=0; j<seq_len; j++) {
                    float score = 0.0f;
                    for (int d=0; d<DK; d++) 
                        score += q[i*DIM + h*DK + d] * k[j*DIM + h*DK + d];
                    score *= scale;
                    if (j > i) score = MASK_VAL; // Causal Mask
                    attn[h*seq_len*seq_len + i*seq_len + j] = score;
                }
                softmax(attn + h*seq_len*seq_len + i*seq_len, seq_len);
            }
        }
        
        // Apply Attention to V
        memset(x_norm, 0, seq_len * DIM * sizeof(float)); // Reuse x_norm as output buffer
        for (int h=0; h<N_HEADS; h++) {
            for (int i=0; i<seq_len; i++) {
                for (int d=0; d<DK; d++) {
                    float val = 0.0f;
                    for (int j=0; j<seq_len; j++)
                        val += attn[h*seq_len*seq_len + i*seq_len + j] * v[j*DIM + h*DK + d];
                    x_norm[i*DIM + h*DK + d] = val;
                }
            }
        }
        // Projection & Residual
        float* proj = (float*)calloc(seq_len * DIM, sizeof(float));
        matmul(x_norm, w_o_all + l*DIM*DIM, b_o_all + l*DIM, proj, seq_len, DIM, DIM);
        for(int i=0; i<seq_len*DIM; i++) x[i] += proj[i];
        free(proj);

        // --- MLP ---
        layer_norm(x, ln_2_g_all + l*DIM, ln_2_b_all + l*DIM, x_norm, seq_len);
        float* hidden = (float*)malloc(seq_len * DFF * sizeof(float));
        matmul(x_norm, w_ff1_all + l*DIM*DFF, b_ff1_all + l*DFF, hidden, seq_len, DIM, DFF);
        for(int i=0; i<seq_len*DFF; i++) hidden[i] = gelu(hidden[i]);
        matmul(hidden, w_ff2_all + l*DFF*DIM, b_ff2_all + l*DIM, x_norm, seq_len, DFF, DIM);
        for(int i=0; i<seq_len*DIM; i++) x[i] += x_norm[i]; // Residual
        free(hidden);
    }

    // 4. Final Norm & Head
    layer_norm(x, ln_f_g, ln_f_b, x, seq_len);
    
    // Predict based on last token state
    float* last_state = x + (seq_len - 1) * DIM;
    float max_logit = -1e9f;
    int best_id = 0;

    for (int v = 0; v < VOCAB; v++) {
        float dot = 0.0f;
        for (int d = 0; d < DIM; d++) dot += last_state[d] * w_emb_word[v * DIM + d];
        if (dot > max_logit) { max_logit = dot; best_id = v; }
    }

    free(x); free(x_norm); free(attn); free(q); free(k); free(v); free(logits);
    return best_id;
}

// --- Text Processing ---

void load_vocabulary() {
    FILE* f = fopen("vocab.txt", "r");
    if(!f) { printf("Error: vocab.txt not found.\n"); exit(1); }
    char line[MAX_TOKEN_LEN];
    int idx = 0;
    while(fgets(line, sizeof(line), f) && idx < VOCAB) {
        line[strcspn(line, "\r\n")] = 0; // Strip newline
        strcpy(vocabulary[idx], line);
        idx++;
    }
    fclose(f);
    printf("Loaded %d words from vocabulary.\n", idx);
}

// Naive Tokenizer: Finds exact string match in vocab
int encode_word(char* word, int prepend_space) {
    if (prepend_space) {
        char with_space[MAX_TOKEN_LEN] = " ";
        strcat(with_space, word);
        for(int i=0; i<VOCAB; i++) {
            if(strcmp(with_space, vocabulary[i]) == 0) return i;
        }
    }
    // Try exact match
    for(int i=0; i<VOCAB; i++) {
        if(strcmp(word, vocabulary[i]) == 0) return i;
    }
    return 50256; // <|endoftext|> or Unknown
}

// --- Main ---

void alloc_and_load() {
    w_emb_word = (float*)malloc(VOCAB * DIM * 4);
    w_emb_pos  = (float*)malloc(MAX_POS * DIM * 4);
    ln_f_g = (float*)malloc(DIM*4); ln_f_b = (float*)malloc(DIM*4);
    
    // Allocate Layers (Simplification: just one big block per type for all layers)
    int sz_attn = N_LAYERS * DIM * DIM;
    int sz_bias = N_LAYERS * DIM;
    w_q_all = (float*)malloc(sz_attn*4); b_q_all = (float*)malloc(sz_bias*4);
    w_k_all = (float*)malloc(sz_attn*4); b_k_all = (float*)malloc(sz_bias*4);
    w_v_all = (float*)malloc(sz_attn*4); b_v_all = (float*)malloc(sz_bias*4);
    w_o_all = (float*)malloc(sz_attn*4); b_o_all = (float*)malloc(sz_bias*4);
    ln_1_g_all = (float*)malloc(sz_bias*4); ln_1_b_all = (float*)malloc(sz_bias*4);
    
    w_ff1_all = (float*)malloc(N_LAYERS * DIM * DFF * 4);
    b_ff1_all = (float*)malloc(N_LAYERS * DFF * 4);
    w_ff2_all = (float*)malloc(N_LAYERS * DFF * DIM * 4);
    b_ff2_all = (float*)malloc(sz_bias*4);
    ln_2_g_all = (float*)malloc(sz_bias*4); ln_2_b_all = (float*)malloc(sz_bias*4);

    FILE* f = fopen("gpt2_tiny.bin", "rb");
    if(!f) { printf("Error: gpt2_tiny.bin not found.\n"); exit(1); }
    fread(w_emb_word, 4, VOCAB*DIM, f);
    fread(w_emb_pos, 4, MAX_POS*DIM, f);
    
    for(int l=0; l<N_LAYERS; l++) {
        fread(w_q_all + l*DIM*DIM, 4, DIM*DIM, f); fread(b_q_all + l*DIM, 4, DIM, f);
        fread(w_k_all + l*DIM*DIM, 4, DIM*DIM, f); fread(b_k_all + l*DIM, 4, DIM, f);
        fread(w_v_all + l*DIM*DIM, 4, DIM*DIM, f); fread(b_v_all + l*DIM, 4, DIM, f);
        fread(w_o_all + l*DIM*DIM, 4, DIM*DIM, f); fread(b_o_all + l*DIM, 4, DIM, f);
        fread(ln_1_g_all + l*DIM, 4, DIM, f);      fread(ln_1_b_all + l*DIM, 4, DIM, f);
        fread(w_ff1_all + l*DIM*DFF, 4, DIM*DFF, f); fread(b_ff1_all + l*DFF, 4, DFF, f);
        fread(w_ff2_all + l*DFF*DIM, 4, DFF*DIM, f); fread(b_ff2_all + l*DIM, 4, DIM, f);
        fread(ln_2_g_all + l*DIM, 4, DIM, f);      fread(ln_2_b_all + l*DIM, 4, DIM, f);
    }
    fread(ln_f_g, 4, DIM, f); fread(ln_f_b, 4, DIM, f);
    fclose(f);
}

int main() {
    load_vocabulary();
    alloc_and_load();

    int input_ids[SEQ_LEN];
    char buffer[256];

    printf("\n=== Tiny GPT-2 Inference Engine ===\n");

    while(1) {
        printf("\nInput text: ");
        if (!fgets(buffer, sizeof(buffer), stdin)) break;
        buffer[strcspn(buffer, "\r\n")] = 0; // Strip newline

        // 1. Naive Tokenization (Split by space)
        int seq_len = 0;
        int is_first = 1;
        char* token = strtok(buffer, " ");
        printf("Tokens: ");
        while(token != NULL && seq_len < SEQ_LEN/2) {
            int id = encode_word(token, !is_first);
            input_ids[seq_len++] = id;
            printf("[%s -> %d] ", token, id);
            is_first = 0;
            token = strtok(NULL, " ");
        }
        printf("\nGenerating: ");

        // 2. Generation Loop
        for (int step = 0; step < 10; step++) {
            if (seq_len >= SEQ_LEN) break;

            int next_id = predict_next_token(input_ids, seq_len);

            printf("%s", vocabulary[next_id]);
            
            input_ids[seq_len++] = next_id;
        }
        printf("\n");
    }

    return 0;
}
