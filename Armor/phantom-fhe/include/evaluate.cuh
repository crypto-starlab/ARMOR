#pragma once

#include <cmath>

#include "ciphertext.h"
#include "context.cuh"
#include "ntt.cuh"
#include "plaintext.h"
#include "prng.cuh"
#include "secretkey.h"
#include "cuda_wrapper.cuh"

namespace phantom {

    size_t
    FindLevelsToDrop(const PhantomContext &context, size_t multiplicativeDepth, double dcrtBits, bool isKeySwitch,
                     bool isAsymmetric);

    __global__ void key_switch_inner_prod_c2_and_evk(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                                     const DModulus *modulus, size_t n, size_t size_QP,
                                                     size_t size_QP_n,
                                                     size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                                     size_t beta, size_t reduction_threshold);

    // used by keyswitch_inplace
    void key_switch_inner_prod(uint64_t *p_cx, const uint64_t *p_t_mod_up, const uint64_t *const *rlk,
                               const phantom::DRNSTool &rns_tool, const DModulus *modulus_QP,
                               size_t reduction_threshold, const cudaStream_t &stream);

    void keyswitch_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
                           const PhantomRelinKey &relin_keys,
                           bool is_relin, // false
                           const cudaStream_t &stream);

/***************************************************** Core APIs ******************************************************/

    // encrypted = -encrypted
    void negate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted);

    inline auto negate(const PhantomContext &context, const PhantomCiphertext &encrypted) {
        PhantomCiphertext destination = encrypted;
        negate_inplace(context, destination);
        return destination;
    }

    // encrypted1 += encrypted2
    void add_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);

    void armor_add_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);

    inline auto
    armor_add(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) {
        PhantomCiphertext destination = encrypted1;
        armor_add_inplace(context, destination, encrypted2);
        return destination;
    }    
    inline auto
    add(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) {
        PhantomCiphertext destination = encrypted1;
        add_inplace(context, destination, encrypted2);
        return destination;
    }

    // destination = encrypteds[0] + encrypteds[1] + ...
    void add_many(const PhantomContext &context, const std::vector<PhantomCiphertext> &encrypteds,
                  PhantomCiphertext &destination);

    // if negate = false (default): encrypted1 -= encrypted2
    // if negate = true: encrypted1 = encrypted2 - encrypted1
    void sub_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                     const bool &negate = false);

    inline auto
    sub(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
        const bool &negate = false) {
        PhantomCiphertext destination = encrypted1;
        sub_inplace(context, destination, encrypted2, negate);
        return destination;
    }

    // encrypted += plain
    void add_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto
    add_plain(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain) {
        PhantomCiphertext destination = encrypted;
        add_plain_inplace(context, destination, plain);
        return destination;
    }

    // encrypted -= plain
    void sub_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto
    sub_plain(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain) {
        PhantomCiphertext destination = encrypted;
        sub_plain_inplace(context, destination, plain);
        return destination;
    }

    // encrypted *= plain
    void
    multiply_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto multiply_plain(const PhantomContext &context, const PhantomCiphertext &encrypted,
                               const PhantomPlaintext &plain) {
        PhantomCiphertext destination = encrypted;
        multiply_plain_inplace(context, destination, plain);
        return destination;
    }
    void
    armor_multiply_plain_inplace(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain, PhantomCiphertext &destination);

    inline auto armor_multiply_plain(const PhantomContext &context, const PhantomCiphertext &encrypted,
                               const PhantomPlaintext &plain) {
        PhantomCiphertext destination;
        armor_multiply_plain_inplace(context, encrypted, plain, destination);
        return destination;
    }
    // encrypted1 *= encrypted2
    void
    multiply_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);

    void
    armor_multiply_inplace(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                             PhantomCiphertext &destination);

    //encrypted1 *= round(delta * scalar)
    void
    multiply_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, double scale, double scalar);   

    inline auto
    armor_multiply(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) {
        PhantomCiphertext destination;;
        armor_multiply_inplace(context, encrypted1, encrypted2, destination);
        return destination;
    }
    inline auto
    multiply(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) {
        PhantomCiphertext destination = encrypted1;
        multiply_inplace(context, destination, encrypted2);
        return destination;
    }

    inline auto
    multiply_scalar(const PhantomContext &context, const PhantomCiphertext &encrypted1, double scale, double scalar){
        PhantomCiphertext destination = encrypted1;
        multiply_inplace(context, destination, scale,  scalar);
        return destination;
    }

    // encrypted1 *= encrypted2
    void multiply_and_relin_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1,
                                    const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys);

    inline auto multiply_and_relin(const PhantomContext &context, const PhantomCiphertext &encrypted1,
                                   const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys) {
        PhantomCiphertext destination = encrypted1;
        multiply_and_relin_inplace(context, destination, encrypted2, relin_keys);
        return destination;
    }

    void relinearize_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                             const PhantomRelinKey &relin_keys);

    inline auto relinearize(const PhantomContext &context, const PhantomCiphertext &encrypted,
                            const PhantomRelinKey &relin_keys) {
        PhantomCiphertext destination = encrypted;
        relinearize_inplace(context, destination, relin_keys);
        return destination;
    }
    void armor_extract_partial_constant_inplace(const PhantomContext &context, const PhantomCiphertext &encrypted, std::vector<uint64_t> &result, bool flag, bool is_version2);
    void armor_decode_constant_inplace(const PhantomContext &context, std::vector<uint64_t>& partial1, std::vector<uint64_t>& partial2, double scale, double scalar, double &result);
    
    void armor_decode_constant_single_inplace(const PhantomContext &context, std::vector<uint64_t>& partial, double scale, double scalar, double &result);
    void armor_extract_constant_single_inplace(const PhantomContext &context, const PhantomCiphertext &encrypted, std::vector<uint64_t> &result);
    
    // ciphertext
    // ciphertext
    [[nodiscard]]
    PhantomCiphertext rescale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted);

    [[nodiscard]]
    int64_t get_chain_index(const PhantomCiphertext &encrypted);

    [[nodiscard]]
    PhantomCiphertext armor_rescale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted);

    [[nodiscard]]
    PhantomCiphertext armor_to_intt(const PhantomContext &context, PhantomCiphertext &encrypted);
    [[nodiscard]]
    PhantomCiphertext armor_to_ntt(const PhantomContext &context, PhantomCiphertext &encrypted);

    [[nodiscard]]
    PhantomCiphertext armor_rescale_to(const PhantomContext &context, PhantomCiphertext &encrypted, size_t chain_index);

    [[nodiscard]]
    void cudastreamsynchronize();

    
    [[nodiscard]]
    std::vector<uint64_t> armor_extract_partial_constant(const PhantomContext &context, const PhantomCiphertext &encrypted, bool is_norm, bool is_version2);
    [[nodiscard]]
    double armor_decode_constant(const PhantomContext &context, std::vector<uint64_t> &partial1, std::vector<uint64_t> &partial2, double scale, double scalar);
    [[nodiscard]]
    uint64_t getCiphertextBits(const PhantomCiphertext &encrypted);
    [[nodiscard]]
    uint64_t getPartialBits(std::vector<uint64_t> &partial);
    [[nodiscard]]
    PhantomCiphertext armor_upscale_and_rescale(const PhantomContext &context, PhantomCiphertext &encrypted);

    [[nodiscard]]
    PhantomCiphertext armor_mod_switch_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted);
    // ciphertext
    inline void rescale_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted) {
        encrypted = rescale_to_next(context, encrypted);
    }

    [[nodiscard]]
    std::vector<uint64_t> extract_constant(const PhantomContext &context, const PhantomCiphertext &encrypted);
    std::vector<uint64_t> extract_constant_optimize_v1(const PhantomContext &context, const PhantomCiphertext &encrypted);

    std::vector<uint64_t> armor_extract_constant_single(const PhantomContext &context, const PhantomCiphertext &encrypted);
    [[nodiscard]]
    double armor_decode_constant_single(const PhantomContext &context, std::vector<uint64_t> &partial, double scale, double scalar);
    // ciphertext
    [[nodiscard]]
    PhantomCiphertext mod_switch_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted);

    // ciphertext
    inline void mod_switch_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted) {
        encrypted = mod_switch_to_next(context, encrypted);
    }

    // ciphertext
    inline auto mod_switch_to(const PhantomContext &context, const PhantomCiphertext &encrypted, size_t chain_index) {
        if (encrypted.chain_index() > chain_index) {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        PhantomCiphertext destination = encrypted;

        while (destination.chain_index() != chain_index) {
            mod_switch_to_next_inplace(context, destination);
        }

        return destination;
    }

    // ciphertext
    inline void mod_switch_to_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t chain_index) {
        if (encrypted.chain_index() > chain_index) {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        while (encrypted.chain_index() != chain_index) {
            mod_switch_to_next_inplace(context, encrypted);
        }
    }

    // plaintext
    void mod_switch_to_next_inplace(const PhantomContext &context, PhantomPlaintext &plain);

    // plaintext
    inline auto mod_switch_to_next(const PhantomContext &context, const PhantomPlaintext &plain) {
        PhantomPlaintext destination = plain;
        mod_switch_to_next_inplace(context, destination);
        return destination;
    }

    // plaintext
    inline void mod_switch_to_inplace(const PhantomContext &context, PhantomPlaintext &plain, size_t chain_index) {
        if (plain.chain_index() > chain_index) {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        while (plain.chain_index() != chain_index) {
            mod_switch_to_next_inplace(context, plain);
        }
    }

    // plaintext
    inline auto mod_switch_to(const PhantomContext &context, const PhantomPlaintext &plain, size_t chain_index) {
        if (plain.chain_index() > chain_index) {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        PhantomPlaintext destination = plain;

        while (destination.chain_index() != chain_index) {
            mod_switch_to_next_inplace(context, destination);
        }

        return destination;
    }

    void apply_galois_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t galois_elt,
                              const PhantomGaloisKey &galois_keys);

    inline auto apply_galois(const PhantomContext &context, const PhantomCiphertext &encrypted, size_t galois_elt,
                             const PhantomGaloisKey &galois_keys) {
        PhantomCiphertext destination = encrypted;
        apply_galois_inplace(context, destination, galois_elt, galois_keys);
        return destination;
    }
    void armor_rotate_ciphertext_without_keyswitching_in_place(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomCiphertext &destination, int step);

    inline auto armor_rotate_ciphertext_without_keyswitching(const PhantomContext &context, const PhantomCiphertext &encrypted, int step) {
        PhantomCiphertext destination;
        armor_rotate_ciphertext_without_keyswitching_in_place(context, encrypted, destination, step);
        return destination;
    }
        
    void rotate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                        const PhantomGaloisKey &galois_key);

    inline auto rotate(const PhantomContext &context, const PhantomCiphertext &encrypted, int step,
                       const PhantomGaloisKey &galois_key) {
        PhantomCiphertext destination = encrypted;
        rotate_inplace(context, destination, step, galois_key);
        return destination;
    }

/*************************************************** Advanced APIs ****************************************************/

    void hoisting_inplace(const PhantomContext &context, PhantomCiphertext &ct, const PhantomGaloisKey &glk,
                          const std::vector<int> &steps);

    inline auto hoisting(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomGaloisKey &glk,
                         const std::vector<int> &steps) {
        PhantomCiphertext destination = encrypted;
        hoisting_inplace(context, destination, glk, steps);
        return destination;
    }
}
