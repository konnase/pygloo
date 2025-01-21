#include <collective.h>
#include <gloo/barrier.h>
#include <gloo/rendezvous/context.h>

namespace pygloo
{

  void barrier(const std::shared_ptr<gloo::rendezvous::Context> &context, uint32_t tag)
  {
    gloo::BarrierOptions opts_(context);

    opts_.setTag(tag);

    gloo::barrier(opts_);
  }
} // namespace pygloo
