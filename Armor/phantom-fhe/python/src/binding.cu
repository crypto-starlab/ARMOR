#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <cstring>

#include "phantom.h"

namespace py = pybind11;

PYBIND11_MODULE(pyPhantom, m) {

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::enum_<phantom::scheme_type>(m, "scheme_type")
            .value("none", phantom::scheme_type::none)
            .value("bgv", phantom::scheme_type::bgv)
            .value("bfv", phantom::scheme_type::bfv)
            .value("ckks", phantom::scheme_type::ckks)
            .export_values();

    py::enum_<phantom::mul_tech_type>(m, "mul_tech_type")
            .value("none", phantom::mul_tech_type::none)
            .value("behz", phantom::mul_tech_type::behz)
            .value("hps", phantom::mul_tech_type::hps)
            .value("hps_overq", phantom::mul_tech_type::hps_overq)
            .value("hps_overq_leveled", phantom::mul_tech_type::hps_overq_leveled)
            .export_values();

    py::enum_<phantom::arith::sec_level_type>(m, "sec_level_type")
            .value("none", phantom::arith::sec_level_type::none)
            .value("tc128", phantom::arith::sec_level_type::tc128)
            .value("tc192", phantom::arith::sec_level_type::tc192)
            .value("tc256", phantom::arith::sec_level_type::tc256)
            .export_values();

    py::class_<phantom::arith::Modulus>(m, "modulus")
            .def(py::init<std::uint64_t>());

    m.def("create_coeff_modulus", &phantom::arith::CoeffModulus::Create);

    m.def("create_plain_modulus", &phantom::arith::PlainModulus::Batching);

    py::class_<phantom::EncryptionParameters>(m, "params")
            .def(py::init<phantom::scheme_type>())
            .def("set_mul_tech", &phantom::EncryptionParameters::set_mul_tech)
            .def("set_poly_modulus_degree", &phantom::EncryptionParameters::set_poly_modulus_degree)
            .def("set_special_modulus_size", &phantom::EncryptionParameters::set_special_modulus_size)
            .def("set_galois_elts", &phantom::EncryptionParameters::set_galois_elts)
            .def("set_coeff_modulus", &phantom::EncryptionParameters::set_coeff_modulus)
            .def("set_plain_modulus", &phantom::EncryptionParameters::set_plain_modulus);

    py::class_<phantom::util::cuda_stream_wrapper>(m, "cuda_stream")
            .def(py::init<>());

    py::class_<PhantomContext>(m, "context")
            .def(py::init<phantom::EncryptionParameters &>());

    py::class_<PhantomSecretKey>(m, "secret_key")
            .def(py::init<>())
            .def(py::init<const PhantomContext &>())
            .def("to_bytes", [](const PhantomSecretKey &key) {
                phantom::cudastreamsynchronize();
                std::ostringstream stream(std::ios::out | std::ios::binary);
                key.save(stream);
                return py::bytes(stream.str());
            })
            .def_static("from_bytes", [](py::bytes value) {
                const std::string data = value;
                if (data.size() < 3 * sizeof(size_t))
                    throw std::invalid_argument("Truncated secret key");
                size_t dims[3];
                std::memcpy(dims, data.data(), sizeof(dims));
                size_t words = (data.size() - sizeof(dims)) / sizeof(uint64_t);
                if ((data.size() - sizeof(dims)) % sizeof(uint64_t))
                    throw std::invalid_argument("Invalid secret key length");
                for (auto dim : dims) {
                    if (!dim || words % dim) throw std::invalid_argument("Invalid secret key dimensions");
                    words /= dim;
                }
                if (words != 1) throw std::invalid_argument("Secret key length mismatch");
                std::istringstream stream(data, std::ios::in | std::ios::binary);
                PhantomSecretKey key;
                key.load(stream);
                return key;
            })
            .def("gen_publickey", &PhantomSecretKey::gen_publickey)
            .def("gen_multiparty_publickey", 
                  py::overload_cast<const PhantomSecretKey&, const PhantomContext&>(
                  &PhantomSecretKey::gen_multiparty_publickey, py::const_), py::arg("s2"), py::arg("context"))
            .def("armor_gen_keys",
                  py::overload_cast<const PhantomContext&>(
                  &PhantomSecretKey::armor_gen_keys, py::const_),
                  py::arg("context"))
.def("armor_gen_keys_single",
                  py::overload_cast<const PhantomContext&>(
                  &PhantomSecretKey::armor_gen_keys_single, py::const_), py::arg("context"))
            .def("gen_relinkey", &PhantomSecretKey::gen_relinkey)
            .def("create_galois_keys", &PhantomSecretKey::create_galois_keys)
            .def("encrypt_symmetric",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomSecretKey::encrypt_symmetric, py::const_), py::arg(), py::arg())
            .def("decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &>(
                         &PhantomSecretKey::decrypt), py::arg(), py::arg())
            .def("decrypt_cipher",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &>(
                         &PhantomSecretKey::decrypt_cipher), py::arg(), py::arg())
            .def("armor_decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, PhantomSecretKey &, size_t>(
                         &PhantomSecretKey::armor_decrypt), py::arg(), py::arg(), py::arg(), py::arg())
            .def("armor_decrypt_cipher",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, PhantomSecretKey &, size_t>(
                         &PhantomSecretKey::armor_decrypt_cipher), py::arg(), py::arg(),  py::arg(), py::arg())
            .def("armor_individual_decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, bool>(
                         &PhantomSecretKey::armor_individual_decrypt), py::arg(), py::arg(), py::arg())
            .def("armor_individual_decrypt_last",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, bool>(
                         &PhantomSecretKey::armor_individual_decrypt_last), py::arg(), py::arg(), py::arg())
            .def("armor_individual_decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, PhantomSecretKey &, size_t>(
                         &PhantomSecretKey::individual_decrypt), py::arg(),  py::arg(), py::arg(), py::arg())
            .def("armor_individual_decrypt_last",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, PhantomSecretKey &, size_t>(
                         &PhantomSecretKey::individual_decrypt_last), py::arg(), py::arg(), py::arg(), py::arg())
            .def("armor_individual_decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, bool, bool, bool>(
                     &PhantomSecretKey::armor_individual_decrypt),
                 py::arg("context"), py::arg("ciphertext"), py::arg("extract_partial_constant"),
                 py::arg("is_norm"), py::arg("is_version2"))
            .def("armor_individual_decrypt_last",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, bool, bool, bool>(
                     &PhantomSecretKey::armor_individual_decrypt_last),
                 py::arg("context"), py::arg("ciphertext"), py::arg("extract_partial_constant"),
                 py::arg("is_norm"), py::arg("is_version2"))
            .def("armor_individual_decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, PhantomSecretKey &, size_t, bool, bool>(
                     &PhantomSecretKey::armor_individual_decrypt),
                 py::arg("context"), py::arg("ciphertext"), py::arg("sk_remain"), py::arg("step"),
                 py::arg("extract_partial_constant"), py::arg("is_version2"))
            .def("armor_individual_decrypt_last",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, PhantomSecretKey &, size_t, bool, bool>(
                     &PhantomSecretKey::armor_individual_decrypt_last),
                 py::arg("context"), py::arg("ciphertext"), py::arg("sk_remain"), py::arg("step"),
                 py::arg("extract_partial_constant"), py::arg("is_version2"))
            //.def("multiparty_decrypt",
            //     py::overload_cast<const PhantomContext &, const PhantomCiphertext &, std::vector<PhantomCiphertext> &>(
            //             &PhantomSecretKey::multiparty_decrypt), py::arg(), py::arg(), py::arg())
            .def("armor_multiparty_decrypt",
                py::overload_cast<const PhantomContext &, std::vector<PhantomCiphertext> &>(
                         &PhantomSecretKey::multiparty_decrypt), py::arg(), py::arg());

    py::class_<PhantomPublicKey>(m, "public_key")
            .def(py::init<>())
            .def("to_bytes", [](const PhantomPublicKey &key) {
                phantom::cudastreamsynchronize();
                std::ostringstream stream(std::ios::out | std::ios::binary);
                key.save(stream);
                return py::bytes(stream.str());
            })
            .def_static("from_bytes", [](py::bytes value) {
                const std::string data = value;
                // Public keys use the same serialized container as ciphertexts.
                const size_t header = 5 * sizeof(size_t) + sizeof(double)
                    + sizeof(uint64_t) + 2 * sizeof(bool);
                if (data.size() < header) throw std::invalid_argument("Truncated public key");
                size_t fields[4];
                std::memcpy(fields, data.data(), sizeof(fields));
                size_t words = (data.size() - header) / sizeof(uint64_t);
                if ((data.size() - header) % sizeof(uint64_t))
                    throw std::invalid_argument("Invalid public key length");
                for (size_t i = 1; i < 4; ++i) {
                    if (!fields[i] || words % fields[i]) throw std::invalid_argument("Invalid public key dimensions");
                    words /= fields[i];
                }
                if (words != 1) throw std::invalid_argument("Public key length mismatch");
                std::istringstream stream(data, std::ios::in | std::ios::binary);
                PhantomPublicKey key;
                key.load(stream);
                return key;
            })
            .def("encrypt_asymmetric",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomPublicKey::encrypt_asymmetric), py::arg(), py::arg())
            .def("armor_encrypt_asymmetric",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomPublicKey::armor_encrypt_asymmetric), py::arg(), py::arg());

    py::class_<PhantomRelinKey>(m, "relin_key")
            .def(py::init<>())
            .def("to_bytes", [](const PhantomRelinKey& key) {
                phantom::cudastreamsynchronize();
                std::ostringstream stream(std::ios::out | std::ios::binary);
                key.save(stream);
                return py::bytes(stream.str());
            })
            .def_static("from_bytes", [](py::bytes value) {
                std::string data = value;
                if (data.size() < sizeof(size_t)) throw std::invalid_argument("Truncated relin key");
                std::istringstream stream(data, std::ios::in | std::ios::binary);
                PhantomRelinKey key;
                key.load(stream);
                if (!stream) throw std::invalid_argument("Invalid relin key");
                return key;
            });

    py::class_<PhantomGaloisKey>(m, "galois_key")
            .def(py::init<>())
            .def("to_bytes", [](const PhantomGaloisKey& key) {
                phantom::cudastreamsynchronize();
                std::ostringstream stream(std::ios::out | std::ios::binary);
                key.save(stream);
                return py::bytes(stream.str());
            })
            .def_static("from_bytes", [](py::bytes value) {
                std::string data = value;
                if (data.size() < sizeof(size_t)) throw std::invalid_argument("Truncated galois key");
                std::istringstream stream(data, std::ios::in | std::ios::binary);
                PhantomGaloisKey key;
                key.load(stream);
                if (!stream) throw std::invalid_argument("Invalid galois key");
                return key;
            });

    m.def("get_elt_from_step", &phantom::util::get_elt_from_step);

    m.def("get_elts_from_steps", &phantom::util::get_elts_from_steps);

    py::class_<PhantomBatchEncoder>(m, "batch_encoder")
            .def(py::init<const PhantomContext &>())
            .def("slot_count", &PhantomBatchEncoder::slot_count)
            .def("encode",
                 py::overload_cast<const PhantomContext &, const std::vector<uint64_t> &>(
                         &PhantomBatchEncoder::encode, py::const_), py::arg(), py::arg())
            .def("decode",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomBatchEncoder::decode, py::const_), py::arg(), py::arg());

    py::class_<PhantomCKKSEncoder>(m, "ckks_encoder")
            .def(py::init<const PhantomContext &>())
            .def("slot_count", &PhantomCKKSEncoder::slot_count)
            .def("encode_complex_vector",
                 py::overload_cast<const PhantomContext &, const std::vector<cuDoubleComplex> &, double, size_t>(
                         &PhantomCKKSEncoder::encode<cuDoubleComplex>),
                 py::arg(), py::arg(), py::arg(), py::arg("chain_index") = 1)
            .def("encode_double_vector",
                 py::overload_cast<const PhantomContext &, const std::vector<double> &, double, size_t>(
                         &PhantomCKKSEncoder::encode<double>),
                 py::arg(), py::arg(), py::arg(),
                 py::arg("chain_index") = 1)
            .def("decode_complex_vector",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomCKKSEncoder::decode<cuDoubleComplex>),
                 py::arg(), py::arg())
            .def("decode_double_vector",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomCKKSEncoder::decode<double>), py::arg(), py::arg())

            .def("armor_coefficient_encode", [](PhantomCKKSEncoder &self, const PhantomContext &context,
                                                py::array_t<double> values, double scale, size_t chain_index) {
                py::buffer_info buf = values.request();
                if (buf.ndim != 1) throw std::runtime_error("Input must be 1-D numpy array");
    
                double *ptr = static_cast<double *>(buf.ptr);
                size_t count = buf.shape[0];
                PhantomPlaintext destination;
                self.armor_coefficient_encode(context, ptr, count, scale, chain_index, destination);
                return destination;}, py::arg("context"), py::arg("values"), py::arg("scale"), py::arg("chain_index") = 0)
            .def("armor_coefficient_encode_ntt", [](PhantomCKKSEncoder &self, const PhantomContext &context,
                                                py::array_t<double> values, double scale, size_t chain_index) {
                py::buffer_info buf = values.request();
                if (buf.ndim != 1) throw std::runtime_error("Input must be 1-D numpy array");
    
                double *ptr = static_cast<double *>(buf.ptr);
                size_t count = buf.shape[0];
                PhantomPlaintext destination;
                self.armor_coefficient_encode_ntt(context, ptr, count, scale, chain_index, destination);
                return destination;}, py::arg("context"), py::arg("values"), py::arg("scale"), py::arg("chain_index") = 0)
            .def("armor_coefficient_decode_and_mult", [](PhantomCKKSEncoder &self, const PhantomContext &context,
                                                const PhantomPlaintext &plaintext, double scalar) {
                size_t count = context.poly_degree_;
    
                auto result_array = py::array_t<double>(count);
                py::buffer_info buf = result_array.request();
                double* ptr = static_cast<double*>(buf.ptr);
                self.armor_coefficient_decode_and_mult(context, plaintext, ptr, scalar);

                return result_array;
            }, py::arg("context"), py::arg("plaintext"), py::arg("scalar"));
 

    py::class_<PhantomPlaintext>(m, "plaintext")
            .def(py::init<>());

    py::class_<PhantomCiphertext>(m, "ciphertext")
            .def(py::init<>())
            .def("to_bytes", [](const PhantomCiphertext &ct) {
                phantom::cudastreamsynchronize();
                std::ostringstream stream(std::ios::out | std::ios::binary);
                ct.save(stream);
                return py::bytes(stream.str());
            })
            .def_static("from_bytes", [](py::bytes value) {
                const std::string data = value;
                const size_t header = 5 * sizeof(size_t) + sizeof(double)
                    + sizeof(uint64_t) + 2 * sizeof(bool);
                if (data.size() < header) throw std::invalid_argument("Truncated ciphertext header");
                size_t fields[4];
                std::memcpy(fields, data.data(), sizeof(fields));
                size_t words = (data.size() - header) / sizeof(uint64_t);
                if ((data.size() - header) % sizeof(uint64_t))
                    throw std::invalid_argument("Invalid ciphertext payload length");
                for (size_t i = 1; i < 4; ++i) {
                    if (!fields[i] || words % fields[i])
                        throw std::invalid_argument("Invalid ciphertext dimensions");
                    words /= fields[i];
                }
                if (words != 1) throw std::invalid_argument("Ciphertext length mismatch");
                std::istringstream stream(data, std::ios::in | std::ios::binary);
                PhantomCiphertext ct;
                ct.load(stream);
                return ct;
            })
            .def("set_scale", &PhantomCiphertext::set_scale);

    m.def("negate", &phantom::negate, py::arg(), py::arg());

    m.def("cudaStreamSychronize", &phantom::cudastreamsynchronize);

    m.def("add", &phantom::add, py::arg(), py::arg(), py::arg());

    m.def("armor_add", &phantom::armor_add, py::arg(), py::arg(), py::arg());

    m.def("add_plain", &phantom::add_plain, py::arg(), py::arg(), py::arg());

    m.def("add_many", &phantom::add_many, py::arg(), py::arg(), py::arg());

    m.def("sub", &phantom::sub, py::arg(), py::arg(), py::arg(), py::arg("negate") = false);

    m.def("sub_plain", &phantom::sub_plain, py::arg(), py::arg(), py::arg());

    m.def("multiply_scalar", &phantom::multiply_scalar, py::arg(), py::arg(), py::arg(), py::arg());
    
    m.def("multiply", &phantom::multiply, py::arg(), py::arg(), py::arg());

    m.def("armor_multiply", &phantom::armor_multiply, py::arg(), py::arg(), py::arg());

    m.def("armor_to_intt", &phantom::armor_to_intt, py::arg(), py::arg());

    m.def("armor_to_ntt", &phantom::armor_to_ntt, py::arg(), py::arg());

    m.def("armor_rescale_to", &phantom::armor_rescale_to, py::arg(), py::arg(), py::arg());

    m.def("armor_rotate_without_key_switching", &phantom::armor_rotate_ciphertext_without_keyswitching, py::arg(), py::arg(), py::arg());

    m.def("multiply_and_relin", &phantom::multiply_and_relin, py::arg(), py::arg(), py::arg(), py::arg());

    m.def("multiply_plain", &phantom::multiply_plain, py::arg(), py::arg(), py::arg());

     m.def("armor_multiply_plain", &phantom::armor_multiply_plain, py::arg(), py::arg(), py::arg());

    m.def("relinearize", &phantom::relinearize, py::arg(), py::arg(), py::arg());

    m.def("rescale_to_next", &phantom::rescale_to_next, py::arg(), py::arg());

    m.def("armor_rescale_to_next", &phantom::armor_rescale_to_next, py::arg(), py::arg());

    m.def("get_chain_index", &phantom::get_chain_index, py::arg());

    m.def("getCiphertextBits", &phantom::getCiphertextBits, py::arg());

    m.def("getPartialBits", &phantom::getPartialBits, py::arg());

    m.def("armor_upscale_and_rescale", &phantom::armor_upscale_and_rescale, py::arg(), py::arg());

    m.def("armor_extract_partial_constant", &phantom::armor_extract_partial_constant, py::arg(), py::arg(), py::arg(), py::arg());

    m.def("armor_decode_constant", &phantom::armor_decode_constant, py::arg(), py::arg(), py::arg(), py::arg(), py::arg());

    m.def("armor_extract_constant", &phantom::armor_extract_constant_single, py::arg(), py::arg());
    m.def("extract_constant", &phantom::extract_constant, py::arg("context"), py::arg("encrypted"));
    // Compatibility alias for existing examples; the C++ implementation is extract_constant.
    m.def("extract_constant_optimize", &phantom::extract_constant, py::arg("context"), py::arg("encrypted"));
    m.def("extract_constant_optimize_v1", &phantom::extract_constant_optimize_v1, py::arg("context"), py::arg("encrypted"));

    m.def("armor_decode_constant_single", &phantom::armor_decode_constant_single, py::arg(), py::arg(), py::arg(), py::arg());


    m.def("armor_mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &>(&phantom::armor_mod_switch_to_next),
          py::arg(), py::arg());

    m.def("mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(&phantom::mod_switch_to_next),
          py::arg(), py::arg());

    m.def("mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &>(&phantom::mod_switch_to_next),
          py::arg(), py::arg());

    m.def("mod_switch_to", py::overload_cast<const PhantomContext &, const PhantomPlaintext &, size_t>(
            &phantom::mod_switch_to), py::arg(), py::arg(), py::arg());

    m.def("mod_switch_to", py::overload_cast<const PhantomContext &, const PhantomCiphertext &, size_t>(
            &phantom::mod_switch_to), py::arg(), py::arg(), py::arg());

    m.def("apply_galois", &phantom::apply_galois, py::arg(), py::arg(), py::arg(), py::arg());

    m.def("rotate", &phantom::rotate, py::arg(), py::arg(), py::arg(), py::arg());

    m.def("hoisting", &phantom::hoisting, py::arg(), py::arg(), py::arg(), py::arg());
}
