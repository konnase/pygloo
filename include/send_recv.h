#include <math.h>
#include <cstring>

#include "gloo/algorithm.h"
#include "gloo/common/logging.h"
#include "gloo/context.h"
#include <gloo/rendezvous/context.h>

namespace pygloo
{

    template <typename T>
    class Sender
    {
    public:
        const std::shared_ptr<gloo::Context> &context_;
        const T *sends_;
        const unsigned long size;
        const int peer;

        ~Sender() {}
        // 禁止拷贝，允许移动（因为 unique_ptr 不可拷贝）
        Sender(const Sender &) = delete;
        Sender &operator=(const Sender &) = delete;
        Sender(Sender &&) = default;
        Sender &operator=(Sender &&) = default;
        Sender(
            const std::shared_ptr<gloo::Context> &context,
            intptr_t sends,
            const unsigned long size,
            const int peer)
            : context_(context),
              size(size),
              peer(peer),
              bytes_(sizeof(T) * size),
              debug_(false),
              sends_(reinterpret_cast<T *>(sends))
        {
            if (debug_)
            {
                std::cout << "init Sender" << std::endl;
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
        }

        void setDebug(bool debug)
        {
            debug_ = debug;
            sendBuf->setDebug(debug);
        }

        void send(unsigned long offset, unsigned long length, unsigned long roffset)
        {
            if (debug_)
            {
                std::cout << "Sending " << bytes_ << " bytes" << std::endl;
            }

            sendBuf->send(offset, length, roffset);
        }

        void waitSend()
        {
            sendBuf->waitSend();
        }

    protected:
        bool debug_;
        const unsigned long bytes_;
        std::unique_ptr<::gloo::transport::Buffer> sendBuf;
    };

    template <typename T>
    class Recver
    {
    public:
        const std::shared_ptr<gloo::Context> &context_;
        T *recvs_;
        const unsigned long size;
        const int peer;

        ~Recver() {}
        // 禁止拷贝，允许移动（因为 unique_ptr 不可拷贝）
        Recver(const Recver &) = delete;
        Recver &operator=(const Recver &) = delete;
        Recver(Recver &&) = default;
        Recver &operator=(Recver &&) = default;
        Recver(
            const std::shared_ptr<gloo::Context> &context,
            intptr_t recvs,
            const unsigned long size,
            const int peer)
            : context_(context),
              size(size),
              peer(peer),
              bytes_(sizeof(T) * size),
              debug_(false),
              recvs_(reinterpret_cast<T *>(recvs))
        {
            auto slot = context_->nextSlot();
            auto &pair = context_->getPair(peer);
            recvBuf = pair->createRecvBuffer(slot, recvs_, bytes_);
        }

        void setDebug(bool debug)
        {
            debug_ = debug;
            recvBuf->setDebug(debug);
        }

        void recv()
        {
            recvBuf->waitRecv();
        }

    protected:
        bool debug_;
        const unsigned long bytes_;
        std::unique_ptr<::gloo::transport::Buffer> recvBuf;
    };

    template <typename T>
    void bindSender(pybind11::module &m, const std::string &type_name)
    {
        using ClassS = Sender<T>;
        pybind11::class_<ClassS, std::unique_ptr<ClassS>>(m, type_name.c_str())
            .def(pybind11::init<const std::shared_ptr<gloo::rendezvous::Context> &,
                                intptr_t,
                                const unsigned long, const int>(),
                 pybind11::arg("context") = nullptr,
                 pybind11::arg("sends") = nullptr,
                 pybind11::arg("size") = 1, pybind11::arg("peer") = 0)
            .def("setDebug", &ClassS::setDebug)
            .def("send", &ClassS::send)
            .def("waitSend", &ClassS::waitSend);
    }

    template <typename T>
    void bindRecver(pybind11::module &m, const std::string &type_name)
    {
        using ClassR = Recver<T>;
        pybind11::class_<ClassR, std::unique_ptr<ClassR>>(m, type_name.c_str())
            .def(pybind11::init<const std::shared_ptr<gloo::rendezvous::Context> &,
                                intptr_t,
                                const unsigned long, const int>(),
                 pybind11::arg("context") = nullptr,
                 pybind11::arg("recvs") = nullptr,
                 pybind11::arg("size") = 1, pybind11::arg("peer") = 0)
            .def("setDebug", &ClassR::setDebug)
            .def("recv", &ClassR::recv);
    }
} // namespace pygloo
