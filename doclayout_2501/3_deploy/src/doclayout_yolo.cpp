#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <icraft-xrt/core/session.h>
#include <icraft_utils.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "postprocess_yolov10.hpp"
#include "yolov10_utils.hpp"

using namespace icraft::xrt;
using namespace icraft::xir;

void calctime(const icraft::xrt::Session &session);

int main(int argc, char *argv[]) {
  try {
    YAML::Node config = YAML::LoadFile(argv[1]);
    // icraft模型部署相关参数配置
    auto imodel = config["imodel"];
    // 仿真上板的jrpath配置
    auto stage = imodel["stage"].as<std::string>();
    auto folderPath = imodel["dir"].as<std::string>();
    auto runBackend = imodel["run_backend"].as<std::string>();
    checkBackend(runBackend);
    auto cudaMode = imodel["cudamode"].as<bool>();
    auto openSpeedmode = imodel["speedmode"].as<bool>();
    auto openCompressFtmp = imodel["compressFtmp"].as<bool>();

    bool mmuMode = true;
    int ocmOption = 4;
    if (runBackend == "buyi")
      mmuMode = imodel["mmuMode"].as<bool>();
    if (runBackend == "zg330")
      ocmOption = imodel["ocm_option"].as<int>();

    auto [JSON_PATH, RAW_PATH] = getJrPath(runBackend, folderPath, stage);
    // URL配置
    auto ip = imodel["ip"].as<std::string>();
    // 可视化配置
    auto show = imodel["show"].as<bool>();
    auto save = imodel["save"].as<bool>();
    // dumpOutputFtmp配置
    bool dump_output = imodel["dump_output"].as<bool>();
    auto dump_format = imodel["dump_format"].as<std::string>();
    auto log_path = imodel["log_path"].as<std::string>();

    // 数据集相关参数配置
    auto dataset = config["dataset"];
    auto imgRoot = dataset["dir"].as<std::string>();
    auto imgList = dataset["list"].as<std::string>();
    auto names_path = dataset["names"].as<std::string>();
    auto resRoot = dataset["res"].as<std::string>();
    checkDir(resRoot);
    auto LABELS = toVector(names_path);
    // 模型自身相关参数配置
    auto param = config["param"];
    auto conf = param["conf"].as<float>();
    auto iou_thresh = param["iou_thresh"].as<float>();
    auto MULTILABEL = param["multilabel"].as<bool>();
    auto fpga_nms = param["fpga_nms"].as<bool>();
    auto N_CLASS = param["number_of_class"].as<int>();
    auto NOH = param["number_of_head"].as<int>();
    auto ANCHORS =
        param["anchors"].as<std::vector<std::vector<std::vector<float>>>>();

    int bbox_info_channel = 4; // 无DFL
    std::vector<int> ori_out_channels = {N_CLASS, bbox_info_channel};
    int parts = ori_out_channels.size(); // parts = 2

    // 加载network
    Network network = loadNetwork(JSON_PATH, RAW_PATH);
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

    // session执行前必须进行apply部署操作
    session.apply();

    auto imkSession = session.sub(1, 2);
    imkSession.enableTimeProfile(true);
    auto icoreSession = session.sub(2, - 2);
    icoreSession.enableTimeProfile(true);
    auto detpostSession = session.sub(-2, -1);
    detpostSession.enableTimeProfile(true);

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
      std::cout << "Image Size: " << img_tensor.dtype()->shape << std::endl;

      dmaInit(runBackend, netinfo.ImageMake_on, img_tensor, device);
      auto imkOutputs = imkSession.forward({ img_tensor });
      std::cout << "imkOutputs size: ";
      for (const auto& t : imkOutputs) {
          std::cout << t.dtype()->shape << " ";
      }
      std::cout << std::endl;
      auto icoreOutputs = icoreSession.forward(imkOutputs);
      std::cout << "icoreOutputs size: ";
      for (const auto& t : icoreOutputs) {
          std::cout << t.dtype()->shape << " ";
      }
      std::cout << std::endl;
      auto outputs = detpostSession.forward(icoreOutputs);
      std::cout << "detpostOutputs size: ";
      for (const auto& t : outputs) {
          std::cout << t.dtype()->shape << " ";
      }
      std::cout << std::endl;
      // -----dumpOutputFtmp-------
      if (dump_output) {
        dumpOutputFtmp(network, outputs, dump_format, log_path);
      }
      if (runBackend != "host")
        device.reset(1);
      // 计时
#if defined(__aarch64__) || defined(_M_ARM64)
      device.reset(1);
      std::cout << "Image make time:" << std::endl;
      calctime(imkSession);
      std::cout << "Icore time:" << std::endl;
      calctime(icoreSession);
      std::cout << "Detpost time:" << std::endl;
      calctime(detpostSession);

#endif

      // default:netinfo.DetPost_on
      if (netinfo.DetPost_on) {
        std::vector<float> normalratio = netinfo.o_scale;
        auto real_out_channels = getReal_out_channels(
            ori_out_channels, netinfo.detpost_bit, N_CLASS);
        auto _norm = set_norm_by_head(NOH, parts, normalratio);
        auto _stride = get_stride(netinfo);
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

/**
 * Copy from v3.7 utils
 * @param session
 */
void calctime(const icraft::xrt::Session &session) {
  const auto network_name = session->network_view.network()->name;
  double total_hard_time = 0;
  double total_time = 0;
  double total_memcpy_time = 0;
  double total_other_time = 0;
  const auto result = session.timeProfileResults();
  for (auto [k, v] : result) {
    auto &[time1, time2, time3, time4] = v;
    total_time += time1;
    total_memcpy_time += time2;
    total_hard_time += time3;
    total_other_time += time4;
  }
  std::cout << "=======TimeProfileResults of " << network_name
            << "=========" << std::endl;
  std::cout << fmt::format("Total_Time: {} ms, Total_MemcpyTime: {} ms , "
                           "Total_HardTime: {} ms , Total_OtherTime: {}ms",
                           total_time, total_memcpy_time, total_hard_time,
                           total_other_time)
            << std::endl;
}
