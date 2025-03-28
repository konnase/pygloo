#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <gloo/config.h>
#include <gloo/context.h>

// #include <transport.h>
#include <collective.h>
#include <rendezvous.h>
#include <send_recv.h>
#include <sstream>

namespace pygloo
{
      bool transport_tcp_available() { return GLOO_HAVE_TRANSPORT_TCP; }
      bool transport_ibverbs_available() { return GLOO_HAVE_TRANSPORT_IBVERBS; }

      bool transport_uv_available() { return GLOO_HAVE_TRANSPORT_UV; }
} // namespace pygloo

PYBIND11_MODULE(pygloo, m)
{
      m.doc() = "binding gloo from c to python"; // optional module docstring

      m.def("transport_tcp_available", &pygloo::transport_tcp_available,
            "transport_tcp_available");

      m.def("transport_ibverbs_available", &pygloo::transport_ibverbs_available,
            "transport_ibverbs_available");

      m.def("transport_uv_available", &pygloo::transport_uv_available,
            "transport_uv_available");

      pybind11::enum_<pygloo::ReduceOp>(m, "ReduceOp", pybind11::arithmetic())
          .value("SUM", pygloo::ReduceOp::SUM)
          .value("PRODUCT", pygloo::ReduceOp::PRODUCT)
          .value("MIN", pygloo::ReduceOp::MIN)
          .value("MAX", pygloo::ReduceOp::MAX)
          .value("BAND", pygloo::ReduceOp::BAND)
          .value("BOR", pygloo::ReduceOp::BOR)
          .value("BXOR", pygloo::ReduceOp::BXOR)
          .value("UNUSED", pygloo::ReduceOp::UNUSED)
          .export_values();

      pybind11::enum_<gloo::detail::AllreduceOptionsImpl::Algorithm>(
          m, "allreduceAlgorithm", pybind11::arithmetic())
          .value("SUM", gloo::detail::AllreduceOptionsImpl::Algorithm::UNSPECIFIED)
          .value("RING", gloo::detail::AllreduceOptionsImpl::Algorithm::RING)
          .value("BCUBE", gloo::detail::AllreduceOptionsImpl::Algorithm::BCUBE)
          .export_values();

      pybind11::enum_<pygloo::glooDataType_t>(m, "glooDataType_t",
                                              pybind11::arithmetic())
          .value("glooInt8", pygloo::glooDataType_t::glooInt8)
          .value("glooUint8", pygloo::glooDataType_t::glooUint8)
          .value("glooInt32", pygloo::glooDataType_t::glooInt32)
          .value("glooUint32", pygloo::glooDataType_t::glooUint32)
          .value("glooInt64", pygloo::glooDataType_t::glooInt64)
          .value("glooUint64", pygloo::glooDataType_t::glooUint64)
          .value("glooFloat16", pygloo::glooDataType_t::glooFloat16)
          .value("glooFloat32", pygloo::glooDataType_t::glooFloat32)
          .value("glooFloat64", pygloo::glooDataType_t::glooFloat64)
          .export_values();

      m.def("allreduce", &pygloo::allreduce_wrapper,
            pybind11::arg("context") = nullptr, pybind11::arg("sendbuf") = nullptr,
            pybind11::arg("recvbuf") = nullptr, pybind11::arg("size") = nullptr,
            pybind11::arg("datatype") = nullptr,
            pybind11::arg("reduceop") = pygloo::ReduceOp::SUM,
            pybind11::arg("algorithm") = gloo::AllreduceOptions::Algorithm::RING,
            pybind11::arg("tag") = 0);

      m.def("allreduce_ring", &pygloo::allreduce_ring_wrapper,
            pybind11::arg("context") = nullptr, pybind11::arg("sendbuf") = nullptr,
            pybind11::arg("size") = nullptr,
            pybind11::arg("datatype") = nullptr);

      // m.def("test_context", &pygloo::context_wrapper,
      //       pybind11::arg("context") = nullptr);

      // m.def("allgather", &pygloo::allgather_wrapper,
      //       pybind11::arg("context") = nullptr, pybind11::arg("sendbuf") = nullptr,
      //       pybind11::arg("recvbuf") = nullptr, pybind11::arg("size") = nullptr,
      //       pybind11::arg("datatype") = nullptr, pybind11::arg("tag") = 0);
      // m.def("allgatherv", &pygloo::allgatherv_wrapper,
      //       pybind11::arg("context") = nullptr, pybind11::arg("sendbuf") = nullptr,
      //       pybind11::arg("recvbuf") = nullptr, pybind11::arg("size") = nullptr,
      //       pybind11::arg("datatype") = nullptr, pybind11::arg("tag") = 0);

      // m.def("reduce", &pygloo::reduce_wrapper, pybind11::arg("context") = nullptr,
      //       pybind11::arg("sendbuf") = nullptr, pybind11::arg("recvbuf") = nullptr,
      //       pybind11::arg("size") = nullptr, pybind11::arg("datatype") = nullptr,
      //       pybind11::arg("reduceop") = pygloo::ReduceOp::SUM,
      //       pybind11::arg("root") = 0, pybind11::arg("tag") = 0);

      // m.def("scatter", &pygloo::scatter_wrapper, pybind11::arg("context") = nullptr,
      //       pybind11::arg("sendbuf") = nullptr, pybind11::arg("recvbuf") = nullptr,
      //       pybind11::arg("size") = nullptr, pybind11::arg("datatype") = nullptr,
      //       pybind11::arg("root") = 0, pybind11::arg("tag") = 0);

      // m.def("gather", &pygloo::gather_wrapper, pybind11::arg("context") = nullptr,
      //       pybind11::arg("sendbuf") = nullptr, pybind11::arg("recvbuf") = nullptr,
      //       pybind11::arg("size") = nullptr, pybind11::arg("datatype") = nullptr,
      //       pybind11::arg("root") = 0, pybind11::arg("tag") = 0);

      m.def("send", &pygloo::send_wrapper, pybind11::arg("context") = nullptr,
            pybind11::arg("sendbuf") = nullptr, pybind11::arg("size") = nullptr,
            pybind11::arg("datatype") = nullptr, pybind11::arg("peer") = nullptr,
            pybind11::arg("tag") = 0);
      m.def("recv", &pygloo::recv_wrapper, pybind11::arg("context") = nullptr,
            pybind11::arg("recvbuf") = nullptr, pybind11::arg("size") = nullptr,
            pybind11::arg("datatype") = nullptr, pybind11::arg("peer") = nullptr,
            pybind11::arg("tag") = 0);

      // m.def("broadcast", &pygloo::broadcast_wrapper,
      //       pybind11::arg("context") = nullptr, pybind11::arg("sendbuf") = nullptr,
      //       pybind11::arg("recvbuf") = nullptr, pybind11::arg("size") = nullptr,
      //       pybind11::arg("datatype") = nullptr, pybind11::arg("root") = 0,
      //       pybind11::arg("tag") = 0);

      // m.def("reduce_scatter", &pygloo::reduce_scatter_wrapper,
      //       pybind11::arg("context") = nullptr, pybind11::arg("sendbuf") = nullptr,
      //       pybind11::arg("recvbuf") = nullptr, pybind11::arg("size") = nullptr,
      //       pybind11::arg("recvElems") = nullptr,
      //       pybind11::arg("datatype") = nullptr,
      //       pybind11::arg("reduceop") = pygloo::ReduceOp::SUM);

      // m.def("barrier", &pygloo::barrier, pybind11::arg("context") = nullptr,
      //       pybind11::arg("tag") = 0);

      pybind11::class_<gloo::Context, std::shared_ptr<gloo::Context>>(m, "Context")
          .def(pybind11::init<int, int, int>(), pybind11::arg("rank") = 0,
               pybind11::arg("size") = 1, pybind11::arg("base") = 2)
          .def_readonly("rank", &gloo::Context::rank)
          .def_readonly("size", &gloo::Context::size)
          .def_readwrite("base", &gloo::Context::base)
          .def("getDevice", &gloo::Context::getDevice)
          //     .def("getPair", &gloo::Context::getPair)
          .def("createUnboundBuffer", &gloo::Context::createUnboundBuffer)
          .def("nextSlot", &gloo::Context::nextSlot)
          .def("closeConnections", &gloo::Context::closeConnections)
          .def("setTimeout", &gloo::Context::setTimeout)
          .def("getTimeout", &gloo::Context::getTimeout);

      // pybind11::class_<pygloo::SendRecver<float>, std::shared_ptr<pygloo::SendRecver<float>>>(m, "SendRecverFloat")
      //     .def(pybind11::init<const std::shared_ptr<gloo::rendezvous::Context> &,
      //                         intptr_t, intptr_t,
      //                         const int, const int>(),
      //          pybind11::arg("context") = nullptr,
      //          pybind11::arg("sends") = nullptr,
      //          pybind11::arg("recvs") = nullptr,
      //          pybind11::arg("size") = 1, pybind11::arg("peer") = 0)
      //     .def("send", &pygloo::SendRecver<float>::send)
      //     .def("recv", &pygloo::SendRecver<float>::recv)
      //     .def("waitSend", &pygloo::SendRecver<float>::waitSend);

      pygloo::bindSender<int8_t>(m, "SenderInt8");
      pygloo::bindSender<uint8_t>(m, "SenderUInt8");
      pygloo::bindSender<int32_t>(m, "SenderInt32");
      pygloo::bindSender<uint32_t>(m, "SenderUInt32");
      pygloo::bindSender<int64_t>(m, "SenderInt64");
      pygloo::bindSender<uint64_t>(m, "SenderUInt64");
      pygloo::bindSender<gloo::float16>(m, "SenderFloat16");
      pygloo::bindSender<float_t>(m, "SenderFloat");
      pygloo::bindSender<double_t>(m, "SenderDouble");

      pygloo::bindRecver<int8_t>(m, "RecverInt8");
      pygloo::bindRecver<uint8_t>(m, "RecverUInt8");
      pygloo::bindRecver<int32_t>(m, "RecverInt32");
      pygloo::bindRecver<uint32_t>(m, "RecverUInt32");
      pygloo::bindRecver<int64_t>(m, "RecverInt64");
      pygloo::bindRecver<uint64_t>(m, "RecverUInt64");
      pygloo::bindRecver<gloo::float16>(m, "RecverFloat16");
      pygloo::bindRecver<float_t>(m, "RecverFloat");
      pygloo::bindRecver<double_t>(m, "RecverDouble");

      pygloo::transport::def_transport_module(m);
      pygloo::rendezvous::def_rendezvous_module(m);
}
