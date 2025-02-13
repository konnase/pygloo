#include "gloo/rendezvous/tcp_store.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <unistd.h> //需要的头文件

int main()
{
    auto rank = getenv("RANK");
    if (!rank)
    {
        rank = "0";
    }
    auto world_size = getenv("WORLD_SIZE");
    if (!world_size)
    {
        world_size = "1";
    }

    bool is_master = false;
    if (strcmp(rank, "0") == 0)
    {
        is_master = true;
    }
    std::cout << "rank: " << rank << ", world_size: " << world_size << ", is_master: " << is_master << std::endl;

    auto tcp_store = std::make_shared<gloo::rendezvous::TCPStore>("127.0.0.1", 12345, 2, is_master);
    auto store = std::static_pointer_cast<gloo::rendezvous::Store>(tcp_store);

    std::vector<char> data = {'a', '-', rank[0]};
    // if (strcmp(rank, "0") == 0)
    // {
    //     store->set(rank, data);
    // }
    store->set(rank, data);

    sleep(5);
    store->wait({"0", "1"});

    auto value0 = store->get("0");
    auto value1 = store->get("1");

    for (auto &c : value0)
    {
        std::cout << c;
    }
    std::cout << std::endl;
    for (auto &c : value1)
    {
        std::cout << c;
    }
    std::cout << std::endl;
    // std::cout << "data: " << std::string(store->get("1").begin(), store->get("1").end()) << std::endl;

    sleep(5);
}