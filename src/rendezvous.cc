#include <gloo/config.h>
#include <rendezvous.h>

#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/store.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/hash_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/tcp_store.h>

#include <iostream>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#if GLOO_USE_REDIS
#include <gloo/rendezvous/redis_store.h>
#include <pybind11/stl.h>
#endif

using namespace gloo;

namespace pygloo
{
  namespace rendezvous
  {

    void def_rendezvous_module(pybind11::module &m)
    {
      pybind11::module rendezvous =
          m.def_submodule("rendezvous", "This is a rendezvous module");

      pybind11::class_<gloo::rendezvous::Context, gloo::Context,
                       std::shared_ptr<gloo::rendezvous::Context>>(rendezvous,
                                                                   "Context")
          .def(pybind11::init<int, int, int>(), pybind11::arg("rank") = 0,
               pybind11::arg("size") = 1, pybind11::arg("base") = 2)
          .def_readonly("rank", &gloo::rendezvous::Context::rank)
          .def_readonly("size", &gloo::rendezvous::Context::size)
          .def_readwrite("base", &gloo::rendezvous::Context::base)
          .def("nextSlot", &gloo::Context::nextSlot)
          .def("connectFullMesh", &gloo::rendezvous::Context::connectFullMesh);

      pybind11::class_<gloo::rendezvous::Store,
                       std::shared_ptr<gloo::rendezvous::Store>>(rendezvous,
                                                                 "Store")
          .def("set", &gloo::rendezvous::Store::set)
          .def("get", &gloo::rendezvous::Store::get);

      pybind11::class_<gloo::rendezvous::FileStore, gloo::rendezvous::Store,
                       std::shared_ptr<gloo::rendezvous::FileStore>>(rendezvous,
                                                                     "FileStore")
          .def(pybind11::init<const std::string &>())
          .def("set", &gloo::rendezvous::FileStore::set)
          .def("get", &gloo::rendezvous::FileStore::get);

      pybind11::class_<gloo::rendezvous::HashStore, gloo::rendezvous::Store,
                       std::shared_ptr<gloo::rendezvous::HashStore>>(rendezvous,
                                                                     "HashStore")
          .def(pybind11::init([]()
                              { return new gloo::rendezvous::HashStore(); }))
          .def("set", &gloo::rendezvous::HashStore::set)
          .def("get", &gloo::rendezvous::HashStore::get);

      pybind11::class_<gloo::rendezvous::PrefixStore, gloo::rendezvous::Store,
                       std::shared_ptr<gloo::rendezvous::PrefixStore>>(
          rendezvous, "PrefixStore")
          .def(pybind11::init<const std::string &, gloo::rendezvous::Store &>())
          .def("set", &gloo::rendezvous::PrefixStore::set)
          .def("get", &gloo::rendezvous::PrefixStore::get);

      pybind11::class_<gloo::rendezvous::TCPStore, gloo::rendezvous::Store,
                       std::shared_ptr<gloo::rendezvous::TCPStore>>(rendezvous,
                                                                    "TCPStore")
          .def(pybind11::init<const std::string &,
                              int, int,
                              bool, int>(),
               pybind11::arg("hostname") = nullptr, pybind11::arg("port") = nullptr,
               pybind11::arg("world_size") = nullptr, pybind11::arg("is_master") = nullptr,
               pybind11::arg("timeout") = 30)
          .def("set", &gloo::rendezvous::TCPStore::set)
          .def("get", &gloo::rendezvous::TCPStore::get);

#if GLOO_USE_REDIS
      class RedisStoreWithAuth : public gloo::rendezvous::RedisStore
      {
      public:
        RedisStoreWithAuth(const std::string &host, int port)
            : gloo::rendezvous::RedisStore(host, port) {};
        using gloo::rendezvous::RedisStore::check;
        using gloo::rendezvous::RedisStore::get;
        using gloo::rendezvous::RedisStore::redis_;
        using gloo::rendezvous::RedisStore::set;
        using gloo::rendezvous::RedisStore::wait;

        void authorize(std::string redis_password)
        {
          void *ptr =
              (redisReply *)redisCommand(redis_, "auth %b", redis_password.c_str(),
                                         (size_t)redis_password.size());

          if (ptr == nullptr)
          {
            GLOO_THROW_IO_EXCEPTION(redis_->errstr);
          }
          redisReply *reply = static_cast<redisReply *>(ptr);
          if (reply->type == REDIS_REPLY_ERROR)
          {
            GLOO_THROW_IO_EXCEPTION("Error: ", reply->str);
          }
          freeReplyObject(reply);
        }

        void delKey(const std::string &key)
        {
          void *ptr = redisCommand(redis_, "del %b", key.c_str(), (size_t)key.size());

          if (ptr == nullptr)
          {
            GLOO_THROW_IO_EXCEPTION(redis_->errstr);
          }
          redisReply *reply = static_cast<redisReply *>(ptr);
          if (reply->type == REDIS_REPLY_ERROR)
          {
            GLOO_THROW_IO_EXCEPTION("Error: ", reply->str);
          }
          freeReplyObject(reply);
        }

        void delKeys(const std::vector<std::string> &keys)
        {
          bool result = check(keys);
          if (!result)
            GLOO_THROW_IO_EXCEPTION("Error: keys not exist");

          std::vector<std::string> args;
          args.push_back("del");
          for (const auto &key : keys)
          {
            args.push_back(key);
          }

          std::vector<const char *> argv;
          std::vector<size_t> argvlen;
          for (const auto &arg : args)
          {
            argv.push_back(arg.c_str());
            argvlen.push_back(arg.length());
          }

          auto argc = argv.size();
          void *ptr = redisCommandArgv(redis_, argc, argv.data(), argvlen.data());

          if (ptr == nullptr)
          {
            GLOO_THROW_IO_EXCEPTION(redis_->errstr);
          }
          redisReply *reply = static_cast<redisReply *>(ptr);
          if (reply->type == REDIS_REPLY_ERROR)
          {
            GLOO_THROW_IO_EXCEPTION("Error: ", reply->str);
          }
          freeReplyObject(reply);
        }
      };

      pybind11::class_<gloo::rendezvous::RedisStore, gloo::rendezvous::Store,
                       std::shared_ptr<gloo::rendezvous::RedisStore>>(rendezvous,
                                                                      "_RedisStore")
          .def(pybind11::init<const std::string &, int>())
          .def("set", &gloo::rendezvous::RedisStore::set)
          .def("get", &gloo::rendezvous::RedisStore::get);

      pybind11::class_<RedisStoreWithAuth, gloo::rendezvous::RedisStore,
                       gloo::rendezvous::Store,
                       std::shared_ptr<RedisStoreWithAuth>>(rendezvous,
                                                            "RedisStore")
          .def(pybind11::init<const std::string &, int>())
          .def("set", &RedisStoreWithAuth::set)
          .def("get", &RedisStoreWithAuth::get)
          .def("authorize", &RedisStoreWithAuth::authorize)
          .def("delKey", &RedisStoreWithAuth::delKey)
          .def("delKeys", &RedisStoreWithAuth::delKeys);
#endif

      class CustomStore : public gloo::rendezvous::Store
      {
      public:
        explicit CustomStore(const pybind11::object &real_store_py_object)
            : real_store_py_object_(real_store_py_object)
        {
        }

        virtual ~CustomStore() {}

        virtual void set(const std::string &key, const std::vector<char> &data) override
        {
          pybind11::str py_key(key.data(), key.size());
          pybind11::bytes py_data(data.data(), data.size());
          auto set_func = real_store_py_object_.attr("set");
          set_func(py_key, py_data);
        }

        virtual std::vector<char> get(const std::string &key) override
        {
          /// Wait until key being ready.
          wait({key});

          pybind11::str py_key(key.data(), key.size());
          auto get_func = real_store_py_object_.attr("get");
          pybind11::bytes data = get_func(py_key);
          std::string ret_str = data;
          std::vector<char> ret(ret_str.data(), ret_str.data() + ret_str.size());
          return ret;
        }

        virtual void wait(const std::vector<std::string> &keys) override
        {
          wait(keys, Store::kDefaultTimeout);
        }

        virtual void wait(const std::vector<std::string> &keys, const std::chrono::milliseconds &timeout) override
        {
          // We now ignore the timeout_ms.

          pybind11::list py_keys = pybind11::cast(keys);
          auto wait_func = real_store_py_object_.attr("wait");
          wait_func(py_keys);
        }

        void delKeys(const std::vector<std::string> &keys)
        {
          pybind11::list py_keys = pybind11::cast(keys);
          auto del_keys_func = real_store_py_object_.attr("del_keys");
          del_keys_func(py_keys);
        }

      protected:
        const pybind11::object real_store_py_object_;
      };

      pybind11::class_<CustomStore, gloo::rendezvous::Store,
                       std::shared_ptr<CustomStore>>(rendezvous, "CustomStore")
          .def(pybind11::init<const pybind11::object &>())
          .def("set", &CustomStore::set)
          .def("get", &CustomStore::get)
          .def("delKeys", &CustomStore::delKeys);
    }
  } // namespace rendezvous
} // namespace pygloo
