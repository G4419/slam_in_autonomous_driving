//
// Created by xiang on 2022/3/18.
//

#include "ch6/g2o_types.h"
#include "ch6/likelihood_filed.h"
#include "ch6/ceres_type.h"

#include <glog/logging.h>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

namespace sad {

void LikelihoodField::SetTargetScan(Scan2d::Ptr scan) {
    target_ = scan;

    // 在target点上生成场函数
    field_ = cv::Mat(1000, 1000, CV_32F, 30.0);

    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        if (scan->ranges[i] < scan->range_min || scan->ranges[i] > scan->range_max) {
            continue;
        }

        double real_angle = scan->angle_min + i * scan->angle_increment;
        double x = scan->ranges[i] * std::cos(real_angle) * resolution_ + 500;
        double y = scan->ranges[i] * std::sin(real_angle) * resolution_ + 500;

        // 在(x,y)附近填入场函数
        for (auto& model_pt : model_) {
            int xx = int(x + model_pt.dx_);
            int yy = int(y + model_pt.dy_);
            if (xx >= 0 && xx < field_.cols && yy >= 0 && yy < field_.rows &&
                field_.at<float>(yy, xx) > model_pt.residual_) {
                field_.at<float>(yy, xx) = model_pt.residual_;
            }
        }
    }
}

void LikelihoodField::BuildModel() {
    const int range = 20;  // 生成多少个像素的模板
    for (int x = -range; x <= range; ++x) {
        for (int y = -range; y <= range; ++y) {
            model_.emplace_back(x, y, std::sqrt((x * x) + (y * y)));
        }
    }
}

void LikelihoodField::SetSourceScan(Scan2d::Ptr scan) { source_ = scan; }

//2d ransac好像有些问题，改用3d
void LikelihoodField::DegradationDetectionByLine(Scan2d::Ptr scan){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->points.resize(scan->ranges.size());
    for (size_t i = 0; i < source_->ranges.size(); ++i) {
            float r = source_->ranges[i];
            float angle = source_->angle_min + i * source_->angle_increment;
            if (r < source_->range_min || r > source_->range_max ||
                angle < source_->angle_min + 30 * M_PI / 180.0 || angle > source_->angle_max - 30 * M_PI / 180.0) {
                cloud->points[i].x = cloud->points[i].y = cloud->points[i].z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            cloud->points[i].x = r * std::cos(angle);
            cloud->points[i].y = r * std::sin(angle);
            cloud->points[i].z = 0;
    }

    pcl::ModelCoefficients::Ptr coefficients1(new pcl::ModelCoefficients);
    pcl::ModelCoefficients::Ptr coefficients2(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // 创建一个Sample Consensus模型，并设置模型类型为线模型
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);  // 设置RANSAC算法的距离阈值
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients1);

    // 使用ExtractIndices来提取除去直线点之外的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);  // 设置为提取负例，即不属于内点的点
    extract.filter(*outlier_cloud);

    seg.setInputCloud(outlier_cloud);
    seg.segment(*inliers, *coefficients2);

    Vec3f norm1(coefficients1->values[0], coefficients1->values[1], coefficients1->values[2]);
    Vec3f norm2(coefficients2->values[0], coefficients2->values[1], coefficients2->values[2]);

    if (norm1.cross(norm2).norm() < 1e-5)
    {
        LOG(INFO) <<  "DegradationDetection!!!";
    }

}

bool LikelihoodField::AlignGaussNewton(SE2& init_pose) {
    int iterations = 10;
    double cost = 0, lastCost = 0;
    SE2 current_pose = init_pose;
    const int min_effect_pts = 20;  // 最小有效点数
    const int image_boarder = 20;   // 预留图像边界

    has_outside_pts_ = false;
    DegradationDetectionByLine(source_);

    for (int iter = 0; iter < iterations; ++iter) {
        Mat3d H = Mat3d::Zero();
        Vec3d b = Vec3d::Zero();
        cost = 0;

        int effective_num = 0;  // 有效点数

        // 遍历source
        for (size_t i = 0; i < source_->ranges.size(); ++i) {
            float r = source_->ranges[i];
            if (r < source_->range_min || r > source_->range_max) {
                continue;
            }

            float angle = source_->angle_min + i * source_->angle_increment;
            if (angle < source_->angle_min + 30 * M_PI / 180.0 || angle > source_->angle_max - 30 * M_PI / 180.0) {
                continue;
            }

            float theta = current_pose.so2().log();
            Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));


            // 在field中的图像坐标
            //使用math_utils.h中的双线性插值函数
            // Vec2i pf = (pw * resolution_ + Vec2d(500, 500)).cast<int>();
            Vec2d pf = pw * resolution_ + Vec2d(500, 500);

            if (pf[0] >= image_boarder && pf[0] < field_.cols - image_boarder && pf[1] >= image_boarder &&
                pf[1] < field_.rows - image_boarder) {
                effective_num++;

                // 图像梯度
                // float dx = 0.5 * (field_.at<float>(pf[1], pf[0] + 1) - field_.at<float>(pf[1], pf[0] - 1));
                // float dy = 0.5 * (field_.at<float>(pf[1] + 1, pf[0]) - field_.at<float>(pf[1] - 1, pf[0]));
                float dx = 0.5 * (math::GetPixelValue<float>(field_, pf[0], pf[1] + 1) - math::GetPixelValue<float>(field_, pf[0], pf[1] - 1));
                float dy = 0.5 * (math::GetPixelValue<float>(field_, pf[0] + 1, pf[1]) - math::GetPixelValue<float>(field_, pf[0] - 1, pf[1]));

                Vec3d J;
                J << resolution_ * dx, resolution_ * dy,
                    -resolution_ * dx * r * std::sin(angle + theta) + resolution_ * dy * r * std::cos(angle + theta);
                H += J * J.transpose();

                // float e = field_.at<float>(pf[1], pf[0]);
                float e = math::GetPixelValue<float>(field_, pf[0], pf[1]);
                b += -J * e;

                cost += e * e;
            } else {
                has_outside_pts_ = true;
            }
        }

        if (effective_num < min_effect_pts) {
            return false;
        }

        // solve for dx
        Vec3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            break;
        }

        cost /= effective_num;
        if (iter > 0 && cost >= lastCost) {
            break;
        }

        // LOG(INFO) << "iter " << iter << " cost = " << cost << ", effect num: " << effective_num;

        current_pose.translation() += dx.head<2>();
        current_pose.so2() = current_pose.so2() * SO2::exp(dx[2]);
        lastCost = cost;
    }

    init_pose = current_pose;
    return true;
}

cv::Mat LikelihoodField::GetFieldImage() {
    cv::Mat image(field_.rows, field_.cols, CV_8UC3);
    for (int x = 0; x < field_.cols; ++x) {
        for (int y = 0; y < field_.rows; ++y) {
            float r = field_.at<float>(y, x) * 255.0 / 30.0;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(uchar(r), uchar(r), uchar(r));
        }
    }

    return image;
}

bool LikelihoodField::IsOutSide(const cv::Mat& field_image, double range, double angle , SE2& current_pose, float resolution){
    Vec2d pw = current_pose * Vec2d(range * std::cos(angle), range * std::sin(angle));
    Vec2i pf = (pw * resolution_ + Vec2d(field_image.rows / 2, field_image.cols / 2)).cast<int>();  // 图像坐标
    const int image_boarder = 10;

    if (pf[0] >= image_boarder && pf[0] < field_image.cols - image_boarder && pf[1] >= image_boarder &&
        pf[1] < field_image.rows - image_boarder) {
        return false;
    } else {
        return true;
    }
}

bool LikelihoodField::AlignCeres(SE2& init_pose) {
    const double range_th = 15.0;  // 不考虑太远的scan，不准
    const double rk_delta = 0.8;
    SE2 current_pose = init_pose;
    double* pose =  current_pose.data();
    has_outside_pts_ = false;
    ceres::Problem problem;

        // 遍历source
    for (size_t i = 0; i < source_->ranges.size(); ++i) {
        float r = source_->ranges[i];
        if (r < source_->range_min || r > source_->range_max) {
            continue;
        }

        if (r > range_th) {
            continue;
        }

        float angle = source_->angle_min + i * source_->angle_increment;
        if (angle < source_->angle_min + 30 * M_PI / 180.0 || angle > source_->angle_max - 30 * M_PI / 180.0) {
            continue;
        }

        if (IsOutSide(field_, r, angle, init_pose, resolution_))
        {
            has_outside_pts_ = true;
            continue;
        }

        problem.AddResidualBlock(ceres_optimazion::CreateLikelihoodCostFunction(field_, r, angle, resolution_),
                new ceres::HuberLoss(rk_delta), pose);
        
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 10;
    options.num_threads = 16;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    init_pose = Sophus::SE2d::exp(Vec3d(pose[0], pose[1], pose[2]));


    LOG(INFO) << "estimated pose: " << init_pose.translation().transpose()
              << ", theta: " << init_pose.so2().log();

    return true;
    
}


bool LikelihoodField::AlignG2O(SE2& init_pose) {
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    auto* v = new VertexSE2();
    v->setId(0);
    v->setEstimate(init_pose);
    optimizer.addVertex(v);

    const double range_th = 15.0;  // 不考虑太远的scan，不准
    const double rk_delta = 0.8;

    has_outside_pts_ = false;
    // 遍历source
    for (size_t i = 0; i < source_->ranges.size(); ++i) {
        float r = source_->ranges[i];
        if (r < source_->range_min || r > source_->range_max) {
            continue;
        }

        if (r > range_th) {
            continue;
        }

        float angle = source_->angle_min + i * source_->angle_increment;
        if (angle < source_->angle_min + 30 * M_PI / 180.0 || angle > source_->angle_max - 30 * M_PI / 180.0) {
            continue;
        }

        auto e = new EdgeSE2LikelihoodFiled(field_, r, angle, resolution_);
        e->setVertex(0, v);

        if (e->IsOutSide()) {
            has_outside_pts_ = true;
            delete e;
            continue;
        }

        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        auto rk = new g2o::RobustKernelHuber;
        rk->setDelta(rk_delta);
        e->setRobustKernel(rk);
        optimizer.addEdge(e);
    }

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    init_pose = v->estimate();
    return true;
}

void LikelihoodField::SetFieldImageFromOccuMap(const cv::Mat& occu_map) {
    const int boarder = 25;
    field_ = cv::Mat(1000, 1000, CV_32F, 30.0);

    for (int x = boarder; x < occu_map.cols - boarder; ++x) {
        for (int y = boarder; y < occu_map.rows - boarder; ++y) {
            if (occu_map.at<uchar>(y, x) < 127) {
                // 在该点生成一个model
                for (auto& model_pt : model_) {
                    int xx = int(x + model_pt.dx_);
                    int yy = int(y + model_pt.dy_);
                    if (xx >= 0 && xx < field_.cols && yy >= 0 && yy < field_.rows &&
                        field_.at<float>(yy, xx) > model_pt.residual_) {
                        field_.at<float>(yy, xx) = model_pt.residual_;
                    }
                }
            }
        }
    }
}

}  // namespace sad