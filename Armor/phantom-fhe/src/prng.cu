
#include <random>
#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

#include "prng.cuh"
#include "host/globals.h"


using namespace phantom::arith;
using namespace phantom::util;

/** Generate Pseudo random number with provided key and nonce
 * @param[out] out The generated random number
 * @param[in] outlen The required length of random number
 * @param[in] nonce The nonce used for PRNG
 * @param[in] key The key (seed) used for PRNG
 * @param[in] keylen The length of PRNG key
 */
void __device__ salsa20_gpu(uint8_t *out, const size_t outlen, const uint64_t nonce, const uint8_t *key,
                            const size_t keylen) {
    int k;
    uint32_t *mem;
    uint32_t x[16], j[16];
    uint64_t blockno = 0;

    for (k = 0; k < outlen; k++)
        out[k] = 0;

    for (k = 0; k < (outlen / 64); k++) {
        j[0] = x[0] = load_littleendian(key + 0);
        j[1] = x[1] = load_littleendian(key + 4);
        j[2] = x[2] = load_littleendian(key + 8);
        j[3] = x[3] = load_littleendian(key + 12);
        j[4] = x[4] = load_littleendian(key + 16);
        j[5] = x[5] = load_littleendian(key + 20);

        j[6] = x[6] = load_littleendian(key + 24);
        j[7] = x[7] = load_littleendian(key + 28);
        j[8] = x[8] = nonce;
        j[9] = x[9] = nonce >> 32;

        j[10] = x[10] = load_littleendian(key + 32);
        j[11] = x[11] = load_littleendian(key + 36);
        j[12] = x[12] = load_littleendian(key + 40);
        j[13] = x[13] = load_littleendian(key + 44);
        j[14] = x[14] = load_littleendian(key + 48);
        j[15] = x[15] = load_littleendian(key + 52);

        for (int i = 20; i > 0; i -= 2) {
            x[4] ^= rotate(x[0] + x[12], 7);
            x[8] ^= rotate(x[4] + x[0], 9);
            x[12] ^= rotate(x[8] + x[4], 13);
            x[0] ^= rotate(x[12] + x[8], 18);
            x[9] ^= rotate(x[5] + x[1], 7);
            x[13] ^= rotate(x[9] + x[5], 9);
            x[1] ^= rotate(x[13] + x[9], 13);
            x[5] ^= rotate(x[1] + x[13], 18);
            x[14] ^= rotate(x[10] + x[6], 7);
            x[2] ^= rotate(x[14] + x[10], 9);
            x[6] ^= rotate(x[2] + x[14], 13);
            x[10] ^= rotate(x[6] + x[2], 18);
            x[3] ^= rotate(x[15] + x[11], 7);
            x[7] ^= rotate(x[3] + x[15], 9);
            x[11] ^= rotate(x[7] + x[3], 13);
            x[15] ^= rotate(x[11] + x[7], 18);
            x[1] ^= rotate(x[0] + x[3], 7);
            x[2] ^= rotate(x[1] + x[0], 9);
            x[3] ^= rotate(x[2] + x[1], 13);
            x[0] ^= rotate(x[3] + x[2], 18);
            x[6] ^= rotate(x[5] + x[4], 7);
            x[7] ^= rotate(x[6] + x[5], 9);
            x[4] ^= rotate(x[7] + x[6], 13);
            x[5] ^= rotate(x[4] + x[7], 18);
            x[11] ^= rotate(x[10] + x[9], 7);
            x[8] ^= rotate(x[11] + x[10], 9);
            x[9] ^= rotate(x[8] + x[11], 13);
            x[10] ^= rotate(x[9] + x[8], 18);
            x[12] ^= rotate(x[15] + x[14], 7);
            x[13] ^= rotate(x[12] + x[15], 9);
            x[14] ^= rotate(x[13] + x[12], 13);
            x[15] ^= rotate(x[14] + x[13], 18);
        }

        x[0] += j[0];
        x[1] += j[1];
        x[2] += j[2];
        x[3] += j[3];
        x[4] += j[4];
        x[5] += j[5];
        x[6] += j[6];
        x[7] += j[7];
        x[8] += j[8];
        x[9] += j[9];
        x[10] += j[10];
        x[11] += j[11];
        x[12] += j[12];
        x[13] += j[13];
        x[14] += j[14];
        x[15] += j[15];

        mem = (unsigned int *) &(out[blockno * 64]);
        *mem ^= x[0];
        mem++;
        *mem ^= x[1];
        mem++;
        *mem ^= x[2];
        mem++;
        *mem ^= x[3];
        mem++;
        *mem ^= x[4];
        mem++;
        *mem ^= x[5];
        mem++;
        *mem ^= x[6];
        mem++;
        *mem ^= x[7];
        mem++;
        *mem ^= x[8];
        mem++;
        *mem ^= x[9];
        mem++;
        *mem ^= x[10];
        mem++;
        *mem ^= x[11];
        mem++;
        *mem ^= x[12];
        mem++;
        *mem ^= x[13];
        mem++;
        *mem ^= x[14];
        mem++;
        *mem ^= x[15];
        blockno++;
    }
}

/** Generate a random ternary poly.
 * Notice: for x^i, the random is same for different coeff modulus.
 * @param[in] prng_seed The PRNG seed
 * @param[in] modulus
 * @param[in] poly_degree The degree of poly
 * @param[in] coeff_mod_size The number of coeff modulus
 */
__global__ void sample_ternary_poly(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                    const uint64_t poly_degree, const uint64_t coeff_mod_size) {
    uint8_t tmp[64];
    uint64_t flag;

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        int twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();

        // The rns compose of all ternary vectors MUST maintain ternary
        // which indicates the ternary vectors of each rns modulus MUST be the same
        // but with different moduli
        // Therefore, tid (the nonce) MUST modulo poly_degree
        salsa20_gpu(tmp, 64, tid % poly_degree, prng_seed, phantom::util::global_variables::prng_seed_byte_count);
        tmp[0] = tmp[0] % 3;
        // if rand=0, flag=0xFFFFFFFF.
        flag = static_cast<uint64_t>(-static_cast<int64_t>(tmp[0] == 0));
        out[tid] = tmp[0] + (flag & mod_value) - 1;
    }
}

/** Generate a random uniform poly
 * Notice: generated random must be less than 0xFFFFFFFFFFFFFFFFULL - modulus - 1 to ensure uniform
 * @param[out] destination The buffer to stored the generated rand
 * @param[in] device_prng_seed The PRNG seed
 * @param[in] poly_degree The degree of poly
 * @param[in] total_size N * coeff_size
 * @param[in] coeff_index The index of the coeff modulus
 * @param[in] coeff_mod The corresponding coeff modulus
 */
__global__ void sample_uniform_poly(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                    const uint64_t poly_degree, const uint64_t coeff_mod_size) {
    uint8_t tmp[64];
    uint64_t *rnd = (uint64_t *) tmp;
    size_t index = 0;
    size_t tries = 0;
    constexpr uint64_t max_random = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (poly_degree >> 3) * coeff_mod_size; // 8 = 2^3, one PRNG invocation generates 64 bytes, i.e, 8 uint64_t.
         tid += blockDim.x * gridDim.x) {
        int twr = tid / (poly_degree >> 3);
        DModulus mod = modulus[twr];

        // sample uniformly from 0 ~ n*mod-1
        uint64_t max_multiple = max_random - barrett_reduce_uint64_uint64(max_random, mod.value(), mod.const_ratio()[1])
                                - 1;

        auto start_pos = tid * 8;
        salsa20_gpu(tmp, 64, tid, prng_seed, phantom::util::global_variables::prng_seed_byte_count);
        tries++;
        while (index < 8) {
            while (rnd[index] > max_multiple) {
                salsa20_gpu(tmp, 64, tid + tries * poly_degree * coeff_mod_size, prng_seed,
                            phantom::util::global_variables::prng_seed_byte_count);
                tries++;
            }
            out[start_pos + index] = barrett_reduce_uint64_uint64(rnd[index], mod.value(), mod.const_ratio()[1]);
            index++;
        }
    }
}

void sample_uniform_poly_wrap(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                              const uint64_t poly_degree, const uint64_t coeff_mod_size, const cudaStream_t &stream) {
    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    sample_uniform_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(out, prng_seed, modulus, poly_degree, coeff_mod_size);
}

/** noise sampling has two methods:
 * 1. rounded Gaussian generation. supprots any std (max derivation = 6 * std), seal uses std::normal_distribution for processing.
 * 2. Centered Binomial Distribution. Only support std <= 3.2 (const global value). Seal defaults uses this.
 * Notice: in this version, we use Centered Binomial Distribution for noise.
 * Notice: for x^i, the random is same for different coeff modulus.
 * @param[in] prng_seed The PRNG seed
 * @param[in] modulus
 * @param[in] poly_degree The degree of poly
 * @param[in] coeff_mod_size The number of coeff modulus
 */
__global__ void sample_error_poly(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                  const uint64_t poly_degree, const uint64_t coeff_mod_size) {
    uint8_t tmp[64];
    int32_t cbd;
    uint64_t flag;

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        int twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();

        salsa20_gpu(tmp, 64, tid % poly_degree, prng_seed, phantom::util::global_variables::prng_seed_byte_count);
        cbd = hamming_weight_uint8(tmp[0]) +
              hamming_weight_uint8(tmp[1]) +
              hamming_weight_uint8(tmp[2] & 0x1F) -
              hamming_weight_uint8(tmp[3]) -
              hamming_weight_uint8(tmp[4]) -
              hamming_weight_uint8(tmp[5] & 0x1F);
        flag = static_cast<uint64_t>(-static_cast<int64_t>(cbd < 0));
        out[tid] = static_cast<uint64_t>(cbd) + (flag & mod_value);
    }
}
__global__ void armor_sample_error_poly_and_add(uint64_t *error, uint64_t *poly, uint64_t *result, const uint8_t *prng_seed, const DModulus *modulus,
                                  const uint64_t poly_degree, const uint64_t coeff_mod_size) {
    uint8_t tmp[64];
    int32_t cbd;
    uint64_t flag;

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        int twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();

        salsa20_gpu(tmp, 64, tid % poly_degree, prng_seed, phantom::util::global_variables::prng_seed_byte_count);
        cbd = hamming_weight_uint8(tmp[0]) +
              hamming_weight_uint8(tmp[1]) +
              hamming_weight_uint8(tmp[2] & 0x1F) -
              hamming_weight_uint8(tmp[3]) -
              hamming_weight_uint8(tmp[4]) -
              hamming_weight_uint8(tmp[5] & 0x1F);
        flag = static_cast<uint64_t>(-static_cast<int64_t>(cbd < 0));
        error[tid] = static_cast<uint64_t>(cbd) + (flag & mod_value);

        result[tid] = add_uint64_uint64_mod(error[tid], poly[tid], mod_value);
    }
}


/*
implement karney dgs kernel

reference :https://github.com/openfheorg/openfhe-development/blob/1306d14f8c26bb6150d3e6ad54f28dfe1007689e/src/core/include/math/discretegaussiangenerator-impl.h#L114
*/
__device__ bool device_algorithm_h(curandState* state) {
    float h_a = curand_uniform(state);
    if (h_a > 0.5f) return true;

    while (true) {
        float h_b = curand_uniform(state);
        if (h_b > h_a) return false;
        else if (h_b < h_a) h_a = curand_uniform(state);
        else return false; 

        if (h_a > h_b) return true;
        else if (h_a == h_b) return true;
    }
}

__device__ bool device_algorithm_b(curandState* state, int32_t k, double x) {
    double y = x;
    int32_t n = 0;
    double m = 2.0 * k + 2.0;

    while (true) {
        double z = curand_uniform_double(state);
        if (z >= y) break;

        double r = curand_uniform_double(state);
        double rTemp = (2.0 * k + x) / m;
        if (r >= rTemp) break;
        
        y = z;
        n++;
    }
    return (n % 2) == 0;
}
__global__ void karney_dgs_kernel(int64_t* error, int poly_degree, double mean, double stddev, uint64_t seed) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree; tid += blockDim.x * gridDim.x){
        if (tid >= poly_degree) return;
        curandState local_state;
        curand_init(seed, tid, 0, &local_state);

        while (true) {
            int32_t k = 0;
            while (device_algorithm_h(&local_state)) k++;

            int32_t p_n = k * (k - 1);
            bool accepted_p = true;
            for (int i = 0; i < p_n; i++) {
                if (!device_algorithm_h(&local_state)) {
                    accepted_p = false;
                    break;
                }
            }
            if (!accepted_p) continue;

            int64_t s = (curand_uniform(&local_state) < 0.5f) ? -1 : 1;
            int64_t j = (int64_t)(curand_uniform_double(&local_state) * std::ceil(stddev));

            double d_i0 = stddev * (double)k + (double)s * mean;
            int64_t i0 = (int64_t)std::ceil(d_i0);
            double x0 = (i0 - d_i0) / stddev;
            double x = x0 + (double)j / stddev;

            if (x >= 1.0 || (x == 0 && s < 0 && k == 0)) continue;

            int32_t h = k + 1;
            bool accepted_b = true;
            while (h--) {
                if (!device_algorithm_b(&local_state, k, x)) {
                    accepted_b = false;
                    break;
                }
            }
            if (accepted_b) {
                error[tid] = s * (i0 + j);
                break; 
            }
        }
    }
}


__forceinline__ __device__ __int128_t sample_j_70bit(curandState* state) {
    uint64_t j_low = ((uint64_t)(curand_uniform_double(state) * 34359738368.0)) & 0x7FFFFFFFFULL;
    uint64_t j_high = ((uint64_t)(curand_uniform_double(state) * 34359738368.0)) & 0x7FFFFFFFFULL;

    __int128_t res = ((__int128_t)j_high << 35) | (__int128_t)j_low;
    return res;
}

__device__ bool device_algorithm_b_fixed(curandState* state, int32_t k, __int128_t j_u, __int128_t i0_offset, __int128_t stddev_int) {
    __int128_t num = i0_offset + j_u;
    __int128_t m = 2 * k + 2;
    __int128_t y_num = num; 
    int32_t n = 0;

    while (true) {
        __int128_t z_num = sample_j_70bit(state);
        
        if (z_num >= y_num) break;

        __int128_t r_num = sample_j_70bit(state);

        if (r_num * m >= (2 * k * stddev_int + num)) break;
        
        y_num = z_num;
        n++;
    }
    return (n % 2) == 0;
}

__global__ void karney_dgs_large_bit_kernel(__int128_t* error, int poly_degree, double mean,
                                            double stddev, uint64_t seed) {
    __int128_t stddev_int = ((__int128_t)1) << 70; 

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree; tid += blockDim.x * gridDim.x) {
        curandState local_state;
        curand_init(seed, tid, 0, &local_state);

        while (true) {
            int32_t k = 0;
            while (device_algorithm_h(&local_state)) k++;

            int32_t p_n = k * (k - 1);
            bool accepted_p = true;
            for (int i = 0; i < p_n; i++) {
                if (!device_algorithm_h(&local_state)) {
                    accepted_p = false;
                    break;
                }
            }
            if (!accepted_p) continue;

            int s = (curand_uniform(&local_state) < 0.5f) ? -1 : 1;

            __int128_t j_u = sample_j_70bit(&local_state);

            double d_i0 = stddev * (double)k + (double)s * mean;
            double ceil_val = ceil(d_i0);
            __int128_t i0 = (__int128_t)ceil_val;

            __int128_t num_offset = (__int128_t)((ceil_val - d_i0) * pow(2.0, 70)); 

            if (num_offset + j_u >= stddev_int || (num_offset + j_u == 0 && s < 0 && k == 0))
                continue;

            int32_t h = k + 1;
            bool accepted_b = true;
            while (h--) {
                if (!device_algorithm_b_fixed(&local_state, k, j_u, num_offset, stddev_int)) {
                    accepted_b = false;
                    break;
                }
            }

            if (accepted_b) {
                __int128_t result = i0 + j_u;
                __int128_t final_val = i0;
                if (s < 0) final_val = -final_val;
                error[tid] = final_val;
                break; 
            }
        }
    }
}