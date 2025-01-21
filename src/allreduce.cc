#include <collective.h>
#include <gloo/allreduce.h>
#include <gloo/reduce.h>
#include <gloo/rendezvous/context.h>

namespace pygloo
{

  template <typename T>
  void allreduce(const std::shared_ptr<gloo::Context> &context, intptr_t sendbuf,
                 intptr_t recvbuf, size_t size, ReduceOp reduceop,
                 gloo::AllreduceOptions::Algorithm algorithm, uint32_t tag)
  {
    std::cout << "############ allreduce ############" << std::endl;
    std::vector<T *> input_ptr{reinterpret_cast<T *>(sendbuf)};
    std::vector<T *> output_ptr{reinterpret_cast<T *>(recvbuf)};

    std::cout << "size: " << size << "input_ptr.size(): " << input_ptr.size() << std::endl;
    for (size_t i = 0; i < size; i++)
    {
      std::cout << "input_ptr: " << input_ptr[0] << ", value: " << input_ptr[0][i] << std::endl;
      // std::cout << "input_ptr: " << input_ptr[1] << ", value: " << input_ptr[1][i] << std::endl;
    }
    // Configure AllreduceOptions struct and call allreduce function
    gloo::AllreduceOptions opts_(context);
    std::cout << "############ after create opts_ ############" << std::endl;
    opts_.setInputs(input_ptr, size);
    std::cout << "############ after set inputs ############" << std::endl;
    opts_.setOutputs(output_ptr, size);
    std::cout << "############ after set outputs ############" << std::endl;
    opts_.setAlgorithm(algorithm);
    std::cout << "############ before set reduce function ############" << std::endl;
    gloo::ReduceOptions::Func fn = toFunction<T>(reduceop);
    opts_.setReduceFunction(fn);
    std::cout << "############ after set reduce function ############" << std::endl;
    opts_.setMaxSegmentSize(128);
    // opts_.setTag(tag);

    for (size_t i = 0; i < size; i++)
    {
      std::cout << "input " << i << ": " << input_ptr[0][i] << std::endl;
      // std::cout << "input " << i << ": " << input_ptr[1][i] << std::endl;
    }
    gloo::allreduce(opts_);
  }

  void allreduce_wrapper(const std::shared_ptr<gloo::rendezvous::Context> &context,
                         intptr_t sendbuf, intptr_t recvbuf, size_t size,
                         glooDataType_t datatype, ReduceOp reduceop,
                         gloo::AllreduceOptions::Algorithm algorithm,
                         uint32_t tag)
  {
    std::cout << "############ allreduce_wrapper ############" << std::endl;
    std::cout << "context: rank: " << context->rank << " , size: " << context->size << " , base: " << context->base
              << ", data size: " << size << std::endl;
    switch (datatype)
    {
    case glooDataType_t::glooInt8:
      allreduce<int8_t>(context, sendbuf, recvbuf, size, reduceop, algorithm,
                        tag);
      break;
    case glooDataType_t::glooUint8:
      allreduce<uint8_t>(context, sendbuf, recvbuf, size, reduceop, algorithm,
                         tag);
      break;
    case glooDataType_t::glooInt32:
      allreduce<int32_t>(context, sendbuf, recvbuf, size, reduceop, algorithm,
                         tag);
      break;
    case glooDataType_t::glooUint32:
      allreduce<uint32_t>(context, sendbuf, recvbuf, size, reduceop, algorithm,
                          tag);
      break;
    case glooDataType_t::glooInt64:
      allreduce<int64_t>(context, sendbuf, recvbuf, size, reduceop, algorithm,
                         tag);
      break;
    case glooDataType_t::glooUint64:
      allreduce<uint64_t>(context, sendbuf, recvbuf, size, reduceop, algorithm,
                          tag);
      break;
    case glooDataType_t::glooFloat16:
      allreduce<gloo::float16>(context, sendbuf, recvbuf, size, reduceop,
                               algorithm, tag);
      break;
    case glooDataType_t::glooFloat32:
      std::cout << "######### allreduce with float32 #########" << std::endl;
      allreduce<float_t>(context, sendbuf, recvbuf, size, reduceop, algorithm,
                         tag);
      break;
    case glooDataType_t::glooFloat64:
      allreduce<double_t>(context, sendbuf, recvbuf, size, reduceop, algorithm,
                          tag);
      break;
    default:
      std::cout << "bad datatype" << std::endl;
      throw std::runtime_error("Unhandled dataType");
    }
  }

  void context_wrapper(const std::shared_ptr<gloo::rendezvous::Context> &context)
  {
    std::cout << "############ context_wrapper ############" << std::endl;
    std::cout << "context: rank: " << context->rank << " , size: " << context->size << " , base: " << context->base << std::endl;
  }
} // namespace pygloo
