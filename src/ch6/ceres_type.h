#ifndef SLAM_IN_AUTO_DRIVING_CERES_OPTIMATION
#define SLAM_IN_AUTO_DRIVING_CERES_OPTIMATION

#include "common/eigen_types.h"
#include "ceres/ceres.h"
#include "thirdparty/sophus/sophus/so3.hpp"
#include <opencv2/core.hpp>

namespace sad {

namespace ceres_optimazion{

class P2PCeres
{
private:
    Vec2d point_, measurement_;
public:
    P2PCeres(Vec2d point, Vec2d measurement) : point_(point), measurement_(measurement) {};
    ~P2PCeres(){};

    template <typename T>
    bool operator()(const T* const pose, T* residual) const {
        Eigen::Matrix<T, 3, 1> pose_tangent;
        pose_tangent << pose[0], pose[1], pose[2];
        Sophus::SE2<T> se2 = Sophus::SE2<T>::exp(pose_tangent);
        
        Eigen::Matrix<T, 2, 1> pw = se2 * point_.cast<T>();
        Eigen::Matrix<T, 2, 1> error = pw - measurement_.cast<T>();
        residual[0] = error[0];
        residual[1] = error[1];
        return true;
    }



};
//为了防止被多个c文件引用导致出错，必须加static
static ceres::CostFunction* CreateP2PCostFunction(const Vec2d point, const Vec2d measurement) {
    return new ceres::AutoDiffCostFunction<P2PCeres, 2, 3>(
        new P2PCeres(point, measurement)
    );
}
//在使用自动求导时，可以将数据转到T类型，不能将数据从T转到其他类型，如果非得转可以自己手动求导或者数值求导,不能使用外部库，
//比如不能使用std::pow()，而需要使用ceres::pow()
// class LikelihoodCeres{
//     public:
//         LikelihoodCeres(const cv::Mat& field_image, double range, double angle, float resolution = 10.0)
//             : field_image_(field_image), range_(range), angle_(angle), resolution_(resolution) {};
//         ~LikelihoodCeres(){};

//         template <typename T>
//         bool operator()(const T* const pose, T* residual) const {
//             Eigen::Matrix<T, 3, 1> pose_tangent;
//             pose_tangent << pose[0], pose[1], pose[2];
//             Sophus::SE2<T> se2 = Sophus::SE2<T>::exp(pose_tangent);
            
//             Eigen::Matrix<T, 2, 1> pw = se2 * Eigen::Matrix<T, 2, 1>(range_ * std::cos(angle_), range_ * std::sin(angle_));
//             Eigen::Matrix<T, 2, 1> pf = pw * T(resolution_) + 
//                 Eigen::Matrix<T, 2, 1>(field_image_.rows / 2, field_image_.cols / 2) - Eigen::Matrix<T, 2, 1>(0.5, 0.5);  // 图像坐标
            

//             if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
//                 pf[1] < field_image_.rows - image_boarder_) {
                
//                 // residual[0] = T(math::GetPixelValue<float>(field_image_, pf[0].a, pf[1].a));需要使用手动求导或者数值求导
//             } else {
//                 residual[0] = T(0);
//                 return false;
//             }
//             return true;
//         }

//     private:
//         const cv::Mat& field_image_;
//         double range_ = 0;
//         double angle_ = 0;
//         float resolution_ = 10.0;
//         inline static const int image_boarder_ = 10;
// };
// static ceres::CostFunction* CreateLikelihoodCostFunction(const cv::Mat& field_image, double range, double angle, float resolution) {
//     return new ceres::AutoDiffCostFunction<LikelihoodCeres, 1, 3>(
//         new LikelihoodCeres(field_image, range, angle, resolution)
//     );
// }
//数值求导,效果不好
// class LikelihoodCeres{
//     public:
//         LikelihoodCeres(const cv::Mat& field_image, double range, double angle, float resolution = 10.0)
//             : field_image_(field_image), range_(range), angle_(angle), resolution_(resolution) {};
//         ~LikelihoodCeres(){};

//         bool operator()(const double* const pose, double* residual) const {
//             Vec3d pose_tangent;
//             pose_tangent << pose[0], pose[1], pose[2];
//             Sophus::SE2d se2 = Sophus::SE2d::exp(pose_tangent);
        
//             Vec2d pw = se2 * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
//             Vec2d pf = pw * resolution_ + 
//                 Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标
        

//             if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
//                 pf[1] < field_image_.rows - image_boarder_) {
            
//                 residual[0] = math::GetPixelValue<float>(field_image_, pf[0], pf[1]);
//             } else {
//                 residual[0] = 0.0;
//                 return false;
//             }
//             return true;
//         }

//     private:
//         const cv::Mat& field_image_;
//         double range_ = 0;
//         double angle_ = 0;
//         float resolution_ = 10.0;
//         inline static const int image_boarder_ = 10;
// };
// static ceres::CostFunction* CreateLikelihoodCostFunction(const cv::Mat& field_image, double range, double angle, float resolution) {
//     return new ceres::NumericDiffCostFunction<LikelihoodCeres, ceres::CENTRAL, 1, 3>(
//         new LikelihoodCeres(field_image, range, angle, resolution)
//     );
// }

// 解析求导
class LikelihoodCeres : public ceres::SizedCostFunction<1, 3> {
    public:
        LikelihoodCeres(const cv::Mat& field_image, double range, double angle, float resolution = 10.0)
            : field_image_(field_image), range_(range), angle_(angle), resolution_(resolution) {};
        ~LikelihoodCeres(){};

        bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override{
            Vec3d pose_tangent;
            pose_tangent << parameters[0][0], parameters[0][1], parameters[0][2];
            Sophus::SE2d se2 = Sophus::SE2d::exp(pose_tangent);
            float theta = se2.so2().log();
            Vec2d pw = se2 * Vec2d(range_ * ceres::cos(angle_), range_ * ceres::sin(angle_));
            Vec2d pf = pw * resolution_ + 
                Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

            if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
                pf[1] < field_image_.rows - image_boarder_) {
                residuals[0] = math::GetPixelValue<float>(field_image_, pf[0], pf[1]);
                // 图像梯度
                float dx = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0] + 1, pf[1]) -
                                math::GetPixelValue<float>(field_image_, pf[0] - 1, pf[1]));
                float dy = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0], pf[1] + 1) -
                                math::GetPixelValue<float>(field_image_, pf[0], pf[1] - 1));
                if (jacobians != nullptr && jacobians[0] != nullptr)
                {
                    jacobians[0][0] = resolution_ * dx;
                    jacobians[0][1] = resolution_ * dy;
                    jacobians[0][2] = -resolution_ * dx * range_ * ceres::sin(angle_ + theta) +
                        resolution_ * dy * range_ * ceres::cos(angle_ + theta);
                }
                
                
            } else {
                residuals[0] = 0.0;
                return false;
            }

            return true;

        }

    private:
        const cv::Mat& field_image_;
        double range_ = 0;
        double angle_ = 0;
        float resolution_ = 10.0;
        inline static const int image_boarder_ = 10;
};
static ceres::CostFunction* CreateLikelihoodCostFunction(const cv::Mat& field_image, double range, double angle, float resolution) {
    return (new LikelihoodCeres(field_image, range, angle, resolution));
}

}

}

#endif
