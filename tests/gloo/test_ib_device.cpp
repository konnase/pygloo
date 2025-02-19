#include <infiniband/verbs.h>
#include <iostream>

bool check_ib_available()
{
    // 获取 InfiniBand 设备列表
    ibv_device **devices = ibv_get_device_list(nullptr);

    if (devices == nullptr)
    {
        std::cerr << "No InfiniBand devices found." << std::endl;
        return false;
    }

    // 打印所有可用的 InfiniBand 设备
    std::cout << "InfiniBand devices found:" << std::endl;
    for (int i = 0; devices[i] != nullptr; ++i)
    {
        std::cout << "Device " << i << ": " << ibv_get_device_name(devices[i]) << std::endl;
    }

    // 清理设备列表
    ibv_free_device_list(devices);

    return true;
}

int main()
{
    if (check_ib_available())
    {
        std::cout << "InfiniBand is available." << std::endl;
    }
    else
    {
        std::cout << "InfiniBand is not available." << std::endl;
    }

    return 0;
}
