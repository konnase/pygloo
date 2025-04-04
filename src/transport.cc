#include <chrono>
#include <transport.h>
#include <infiniband/verbs.h>
#include <sys/stat.h>
#include <iostream>

namespace pygloo
{
  namespace transport
  {

#if GLOO_HAVE_TRANSPORT_TCP
    template <typename... Args>
    using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

    void def_transport_tcp_module(pybind11::module &m)
    {
      pybind11::module tcp = m.def_submodule("tcp", "This is a tcp module");

      tcp.def("CreateDevice", &gloo::transport::tcp::CreateDevice);

      pybind11::class_<gloo::transport::tcp::attr>(tcp, "attr")
          .def(pybind11::init<>())
          .def(pybind11::init<const char *>())
          .def_readwrite("hostname", &gloo::transport::tcp::attr::hostname)
          .def_readwrite("iface", &gloo::transport::tcp::attr::iface)
          .def_readwrite("ai_family", &gloo::transport::tcp::attr::ai_family)
          .def_readwrite("hostname", &gloo::transport::tcp::attr::hostname)
          .def_readwrite("ai_socktype", &gloo::transport::tcp::attr::ai_socktype)
          .def_readwrite("ai_protocol", &gloo::transport::tcp::attr::ai_protocol)
          .def_readwrite("ai_addr", &gloo::transport::tcp::attr::ai_addr)
          .def_readwrite("ai_addrlen", &gloo::transport::tcp::attr::ai_addrlen);

      pybind11::class_<gloo::transport::tcp::Context,
                       std::shared_ptr<gloo::transport::tcp::Context>>(tcp,
                                                                       "Context")
          .def(pybind11::init<std::shared_ptr<gloo::transport::tcp::Device>, int,
                              int>())
          // .def("createPair", &gloo::transport::tcp::Context::createPair)
          .def("createUnboundBuffer",
               &gloo::transport::tcp::Context::createUnboundBuffer);

      pybind11::class_<gloo::transport::tcp::Device,
                       std::shared_ptr<gloo::transport::tcp::Device>,
                       gloo::transport::Device>(tcp, "Device")
          .def(pybind11::init<const struct gloo::transport::tcp::attr &>());
    }
#else
    void def_transport_tcp_module(pybind11::module &m)
    {
      pybind11::module tcp = m.def_submodule("tcp", "This is a tcp module");
    }
#endif

#if GLOO_HAVE_TRANSPORT_UV
    void def_transport_uv_module(pybind11::module &m)
    {
      pybind11::module uv = m.def_submodule("uv", "This is a uv module");

      uv.def("CreateDevice", &gloo::transport::uv::CreateDevice, "CreateDevice");

      pybind11::class_<gloo::transport::uv::attr>(uv, "attr")
          .def(pybind11::init<>())
          .def(pybind11::init<const char *>())
          .def_readwrite("hostname", &gloo::transport::uv::attr::hostname)
          .def_readwrite("iface", &gloo::transport::uv::attr::iface)
          .def_readwrite("ai_family", &gloo::transport::uv::attr::ai_family)
          .def_readwrite("ai_socktype", &gloo::transport::uv::attr::ai_socktype)
          .def_readwrite("ai_protocol", &gloo::transport::uv::attr::ai_protocol)
          .def_readwrite("ai_addr", &gloo::transport::uv::attr::ai_addr)
          .def_readwrite("ai_addrlen", &gloo::transport::uv::attr::ai_addrlen);

      pybind11::class_<gloo::transport::uv::Context,
                       std::shared_ptr<gloo::transport::uv::Context>>(uv, "Context")
          .def(pybind11::init<std::shared_ptr<gloo::transport::uv::Device>, int,
                              int>())
          .def("createUnboundBuffer",
               &gloo::transport::uv::Context::createUnboundBuffer);

      pybind11::class_<gloo::transport::uv::Device,
                       std::shared_ptr<gloo::transport::uv::Device>,
                       gloo::transport::Device>(uv, "Device")
          .def(pybind11::init<const struct gloo::transport::uv::attr &>());
    }
#else
    void def_transport_uv_module(pybind11::module &m)
    {
      pybind11::module uv = m.def_submodule("uv", "This is a uv module");
    }
#endif

#if GLOO_HAVE_TRANSPORT_IBVERBS
    template <typename... Args>
    using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

    void def_transport_ibverbs_module(pybind11::module &m)
    {
      pybind11::module ibverbs = m.def_submodule("ibverbs", "This is a ibverbs module");

      ibverbs.def("CreateDevice", &gloo::transport::ibverbs::CreateDevice);
      ibverbs.def("is_available", &check_ib_available);

      pybind11::class_<gloo::transport::ibverbs::attr>(ibverbs, "attr")
          .def(pybind11::init<>())
          .def(pybind11::init<const char *>())
          .def(pybind11::init<const char *, int, int>())
          .def_readwrite("name", &gloo::transport::ibverbs::attr::name)
          .def_readwrite("port", &gloo::transport::ibverbs::attr::port)
          .def_readwrite("index", &gloo::transport::ibverbs::attr::index);

      pybind11::class_<gloo::transport::ibverbs::Context,
                       std::shared_ptr<gloo::transport::ibverbs::Context>>(ibverbs,
                                                                           "Context")
          .def(pybind11::init<std::shared_ptr<gloo::transport::ibverbs::Device>, int,
                              int>())
          // .def("createPair", &gloo::transport::ibverbs::Context::createPair)
          .def("createUnboundBuffer",
               &gloo::transport::ibverbs::Context::createUnboundBuffer);

      pybind11::class_<gloo::transport::ibverbs::Device,
                       std::shared_ptr<gloo::transport::ibverbs::Device>,
                       gloo::transport::Device>(ibverbs, "Device")
          .def(pybind11::init<const struct gloo::transport::ibverbs::attr &, ibv_context *>());
    }

    bool check_ib_available()
    {
      // check /dev/infiniband exists
      struct stat st;
      if (stat("/dev/infiniband", &st) != 0 || !S_ISDIR(st.st_mode))
      {
        return false;
      }

      // check device list
      ibv_device **dev_list = ibv_get_device_list(nullptr);
      if (!dev_list)
      {
        return false;
      }

      bool available = false;
      for (int i = 0; dev_list[i]; ++i)
      {
        ibv_context *ctx = ibv_open_device(dev_list[i]);
        if (!ctx)
        {
          continue;
        }

        // 可选：进一步验证设备属性
        ibv_device_attr attr;
        if (ibv_query_device(ctx, &attr) == 0)
        {
          available = true;
          ibv_close_device(ctx);
          break;
        }
        ibv_close_device(ctx);
      }

      ibv_free_device_list(dev_list);
      return available;
    }
#else
    void def_transport_ibverbs_module(pybind11::module &m)
    {
      pybind11::module ibverbs = m.def_submodule("ibverbs", "This is a ibverbs module");
    }
#endif

    void def_transport_module(pybind11::module &m)
    {
      pybind11::module transport =
          m.def_submodule("transport", "This is a transport module");

      pybind11::class_<gloo::transport::Device,
                       std::shared_ptr<gloo::transport::Device>,
                       pygloo::transport::PyDevice>(transport, "Device",
                                                    pybind11::module_local())
          .def("str", &gloo::transport::Device::str)
          .def("getPCIBusID", &gloo::transport::Device::getPCIBusID)
          .def("getInterfaceSpeed", &gloo::transport::Device::getInterfaceSpeed)
          .def("hasGPUDirect", &gloo::transport::Device::hasGPUDirect)
          .def("createContext", &gloo::transport::Device::createContext);

      def_transport_uv_module(transport);
      def_transport_tcp_module(transport);
      def_transport_ibverbs_module(transport);
    }
  } // namespace transport
} // namespace pygloo
