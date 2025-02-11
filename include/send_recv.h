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

        ~SendRecver() {}
        // 禁止拷贝，允许移动（因为 unique_ptr 不可拷贝）
        SendRecver(const SendRecver &) = delete;
        SendRecver &operator=(const SendRecver &) = delete;
        SendRecver(SendRecver &&) = default;
        SendRecver &operator=(SendRecver &&) = default;
        SendRecver(
            const std::shared_ptr<gloo::Context> &context,
            intptr_t sends,
            intptr_t recvs,
            const int size,
            const int peer)
            : context_(context),
              size(size),
              peer(peer),
              bytes_(sizeof(T) * size),
              debug_(false),
              sends_(reinterpret_cast<T *>(sends)),
              recvs_(reinterpret_cast<T *>(recvs))
        {
            if (debug_)
            {
                std::cout << "init SendRecver" << std::endl;
                std::cout << "rank: " << context_->rank << " world size: " << context_->size << std::endl;
                std::cout << "size: " << size << " peer: " << peer << std::endl;

                std::cout << "initial send " << bytes_ << " bytes" << std::endl;
                for (int i = 0; i < size; ++i)
                {
                    std::cout << sends_[i] << " ";
                }
                std::cout << std::endl;
            }

            auto slot = context_->nextSlot();
            auto &pair = context_->getPair(peer);
            auto sends_temp = const_cast<T *>(sends_);
            sendBuf = pair->createSendBuffer(slot, sends_temp, bytes_);
            recvBuf = pair->createRecvBuffer(slot, recvs_, bytes_);
        }

        void setDebug(bool debug)
        {
            debug_ = debug;
            sendBuf->setDebug(debug);
            recvBuf->setDebug(debug);
        }

        void send()
        {
            if (debug_)
            {
                std::cout << "Sending " << bytes_ << " bytes" << std::endl;
            }

            sendBuf->send();
        }

        void recv()
        {
            recvBuf->waitRecv();
        }

        void waitSend()
        {
            sendBuf->waitSend();
        }

    protected:
        bool debug_;
        const int bytes_;
        std::unique_ptr<::gloo::transport::Buffer> sendBuf;
        std::unique_ptr<::gloo::transport::Buffer> recvBuf;
    };

    template <typename T>
    void bindSendRecver(pybind11::module &m, const std::string &type_name)
    {
        using Class = SendRecver<T>;
        pybind11::class_<Class, std::unique_ptr<Class>>(m, type_name.c_str())
            .def(pybind11::init<const std::shared_ptr<gloo::rendezvous::Context> &,
                                intptr_t, intptr_t,
                                const int, const int>(),
                 pybind11::arg("context") = nullptr,
                 pybind11::arg("sends") = nullptr, pybind11::arg("recvs") = nullptr,
                 pybind11::arg("size") = 1, pybind11::arg("peer") = 0)
            .def("setDebug", &Class::setDebug)
            .def("send", &Class::send)
            .def("recv", &Class::recv)
            .def("waitSend", &Class::waitSend);
    }
} // namespace pygloo
