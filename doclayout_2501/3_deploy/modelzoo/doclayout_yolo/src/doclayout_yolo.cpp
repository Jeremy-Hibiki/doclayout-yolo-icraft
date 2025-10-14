
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/utils.h>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-xrt/dev/host_device.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "icraft_utils.hpp"
#include "postprocess_yolov10.hpp"
#include "yolov10_utils.hpp"

using namespace icraft::xrt;
using namespace icraft::xir;

int main(int argc, char *argv[]) {
  try {
    YAML::Node config = YAML::LoadFile(argv[1]);
    // icraft模型部署相关参数配置
    auto imodel = config["imodel"];
    // 仿真上板的jrpath配置
    std::string stage = imodel["stage"].as<std::string>();
    std::string folderPath = imodel["dir"].as<std::string>();
    std::string runBackend = imodel["run_backend"].as<std::string>();
    checkBackend(runBackend);
    bool cudaMode = imodel["cudamode"].as<bool>();
    bool openSpeedmode = imodel["speedmode"].as<bool>();
    bool openCompressFtmp = imodel["compressFtmp"].as<bool>();

    bool mmuMode = true;
    int ocmOption = 4;
    if (runBackend.compare("buyi") == 0)
      mmuMode = imodel["mmuMode"].as<bool>();
    if (runBackend.compare("zg330") == 0)
      ocmOption = imodel["ocm_option"].as<int>();

    auto JR_PATH = getJrPath(runBackend, folderPath, stage);
    // URL配置
    std::string ip = imodel["ip"].as<std::string>();
    // 可视化配置
    bool show = imodel["show"].as<bool>();
    bool save = imodel["save"].as<bool>();
    // dumpOutputFtmp配置
    bool dump_output = imodel["dump_output"].as<bool>();
    std::string dump_format = imodel["dump_format"].as<std::string>();
    std::string log_path = imodel["log_path"].as<std::string>();

    // 数据集相关参数配置
    auto dataset = config["dataset"];
    std::string imgRoot = dataset["dir"].as<std::string>();
    std::string imgList = dataset["list"].as<std::string>();
    std::string names_path = dataset["names"].as<std::string>();
    std::string resRoot = dataset["res"].as<std::string>();
    checkDir(resRoot);
    auto LABELS = toVector(names_path);
    // 模型自身相关参数配置
    auto param = config["param"];
    float conf = param["conf"].as<float>();
    float iou_thresh = param["iou_thresh"].as<float>();
    bool MULTILABEL = param["multilabel"].as<bool>();
    bool fpga_nms = param["fpga_nms"].as<bool>();
    int N_CLASS = param["number_of_class"].as<int>();
    int NOH = param["number_of_head"].as<int>();
    std::vector<std::vector<std::vector<float>>> ANCHORS =
        param["anchors"].as<std::vector<std::vector<std::vector<float>>>>();

    int bbox_info_channel = 4; // 无DFL
    std::vector<int> ori_out_channels = {N_CLASS, bbox_info_channel};
    int parts = ori_out_channels.size(); // parts = 2

    // 加载network
    Network network = loadNetwork(JR_PATH.first, JR_PATH.second);
    // 初始化netinfo
    NetInfo netinfo = NetInfo(network);

    if (netinfo.DetPost_on) {
      // 更新detpost conf
      updateDetpost(netinfo, conf);
    }

    // 打开device
    Device device =
        openDevice(runBackend, ip, netinfo.mmu || mmuMode, cudaMode);
    // 初始化session
    Session session =
        initSession(runBackend, network, device, ocmOption,
                    netinfo.mmu || mmuMode, openSpeedmode, openCompressFtmp);

    // 开启计时功能
    session.enableTimeProfile(true);
    // session执行前必须进行apply部署操作
    session.apply();

    // 统计图片数量
    int index = 0;
    auto namevector = toVector(imgList);
    int totalnum = namevector.size();
    for (auto name : namevector) {
      progress(index, totalnum);
      index++;
      std::string img_path = imgRoot + '/' + name;
      // 前处理
      PicPre img(img_path, cv::IMREAD_COLOR);

      img.Resize({netinfo.i_cubic[0].h, netinfo.i_cubic[0].w},
                 PicPre::LONG_SIDE)
          .rPad();

      Tensor img_tensor = CvMat2Tensor(img.dst_img, network);

      dmaInit(runBackend, netinfo.ImageMake_on, img_tensor, device);

      std::vector<Tensor> outputs = session.forward({img_tensor});
      // check outputs
      // for(auto output : outputs){
      // 	std::cout << output.dtype()->shape << std::endl;
      // }
      // -----dumpOutputFtmp-------
      if (dump_output) {
        dumpOutputFtmp(network, outputs, dump_format, log_path);
      }
      if (runBackend.compare("host") != 0)
        device.reset(1);
// 计时
// #ifdef __linux__
#if defined(__aarch64__) || defined(_M_ARM64)
      device.reset(1);
      calctime_detail(session);
#endif

      // default:netinfo.DetPost_on
      if (netinfo.DetPost_on) {
        std::vector<float> normalratio = netinfo.o_scale;
        std::vector<int> real_out_channels = _getReal_out_channels(
            ori_out_channels, netinfo.detpost_bit, N_CLASS);
        std::vector<std::vector<float>> _norm =
            set_norm_by_head(NOH, parts, normalratio);
        std::vector<float> _stride = get_stride(netinfo);
        post_detpost_hard(outputs, img, netinfo, conf, iou_thresh, MULTILABEL,
                          fpga_nms, N_CLASS, ANCHORS, LABELS, show, save,
                          resRoot, name, device, runBackend, _norm,
                          real_out_channels, _stride, bbox_info_channel);

      } else {
        post_detpost_soft(outputs, img, LABELS, ANCHORS, netinfo, N_CLASS, conf,
                          iou_thresh, fpga_nms, device, runBackend, MULTILABEL,
                          show, save, resRoot, name);
      }
    }
    // 关闭设备
    Device::Close(device);
    return 0;
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  }
}
