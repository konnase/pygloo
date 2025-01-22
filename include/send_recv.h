#include <math.h>
#include <cstring>

#include "gloo/algorithm.h"
#include "gloo/common/logging.h"
#include "gloo/context.h"
#include <gloo/rendezvous/context.h>

namespace pygloo
{
    template <typename T>
    class SendRecver
    {
    public:
        const std::shared_ptr<gloo::Context> &context_;
        const T *sends_;
        T *recvs_;
        const int size;
        const int peer;
        explicit SendRecver(
            const std::shared_ptr<gloo::Context> &context,
            intptr_t sends,
            intptr_t recvs,
            const int size,
            const int peer)
            : context_(context),
              size(size),
              peer(peer),
              bytes_(sizeof(T) * size)
        {
            sends_ = reinterpret_cast<T *>(sends);
            recvs_ = reinterpret_cast<T *>(recvs);
            std::cout << "init SendRecver" << std::endl;
            std::cout << "rank: " << context_->rank << " world size: " << context_->size << std::endl;
            std::cout << "size: " << size << " peer: " << peer << std::endl;

            std::cout << "initial send " << bytes_ << " bytes" << std::endl;
            for (int i = 0; i < size; ++i)
            {
                std::cout << sends_[i] << " ";
            }
            std::cout << std::endl;

            inbox_ = static_cast<float *>(malloc(bytes_));
            outbox_ = static_cast<float *>(malloc(bytes_));
            auto slot = context_->nextSlot();
            auto &pair = context_->getPair(peer);
            sendBuf = pair->createSendBuffer(slot, outbox_, bytes_);
            recvBuf = pair->createRecvBuffer(slot, inbox_, bytes_);
        }

        void send()
        {
            std::cout << "Sending " << bytes_ << " bytes" << std::endl;
            for (int i = 0; i < size; ++i)
            {
                std::cout << sends_[i] << " ";
            }
            std::cout << std::endl;
            std::memcpy(outbox_, sends_, bytes_);
            sendBuf->send();
        }

        void recv()
        {
            recvBuf->waitRecv();
            std::memcpy(recvs_, inbox_, bytes_);
        }

        void waitSend()
        {
            sendBuf->waitSend();
        }

    protected:
        const int bytes_;
        float *inbox_;
        float *outbox_;
        std::unique_ptr<::gloo::transport::Buffer> sendBuf;
        std::unique_ptr<::gloo::transport::Buffer> recvBuf;
    };

    template <typename T>
    void bindSendRecver(pybind11::module &m, const std::string &type_name)
    {
        using Class = SendRecver<T>;
        pybind11::class_<Class, std::shared_ptr<Class>>(m, type_name.c_str())
            .def(pybind11::init<const std::shared_ptr<gloo::rendezvous::Context> &,
                                intptr_t, intptr_t,
                                const int, const int>(),
                 pybind11::arg("context") = nullptr,
                 pybind11::arg("sends") = nullptr, pybind11::arg("recvs") = nullptr,
                 pybind11::arg("size") = 1, pybind11::arg("peer") = 0)
            .def("send", &Class::send)
            .def("recv", &Class::recv)
            .def("waitSend", &Class::waitSend);
    }

    // template <typename T>
    // std::shared_ptr<SendRecver<T>> newSendRecver(
    //     const std::shared_ptr<gloo::Context> &context,
    //     const T *sends,
    //     T *recvs,
    //     const int size,
    //     const int peer)
    // {
    //     auto sr = SendRecver<T>(context, sends, recvs, size, peer);
    //     return std::make_shared<SendRecver<T>>(sr);
    // };

    // template <typename T>
    // std::shared_ptr<SendRecver<T>> new_send_recver_wrapper(const std::shared_ptr<gloo::rendezvous::Context> &context,
    //                                                        intptr_t sendbuf, intptr_t recvbuf, size_t size, size_t peer,
    //                                                        glooDataType_t datatype);
} // namespace pygloo
