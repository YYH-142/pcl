#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <Eigen/Dense>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// 计算一组点的质心
Eigen::Vector3f computeCentroid(const std::vector<pcl::PointXYZ>& points)
{
    Eigen::Vector3f centroid(0, 0, 0);
    for (const auto& point : points)
    {
        centroid += Eigen::Vector3f(point.x, point.y, point.z);
    }
    centroid /= static_cast<float>(points.size());
    return centroid;
}

// 计算法线之间的夹角（弧度）
float computeAngleBetweenNormals(const Eigen::Vector3f& normal1, const Eigen::Vector3f& normal2)
{
    if (normal1.dot(normal2) / (normal1.norm() * normal2.norm()) > 1)
    {
        return 0;
    }
    return std::acos(normal1.dot(normal2) / (normal1.norm() * normal2.norm()));
}

// 计算标准差
float computeStandardDeviation(const std::vector<float>& values)
{
    if (values.empty()) return 0.0f;

    float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
    float variance = 0.0f;

    for (const auto& value : values)
    {
        variance += (value - mean) * (value - mean);
    }

    variance /= values.size();
    return std::sqrt(variance);
}

int main(int argc, char** argv)
{
    // 创建点云数据类型
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // 读取PCD文件
    const char* path = "D:\\pcd_model\\fandisk-normdirectionnoise0.25.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file %s\n", path);
        return -1;
    }
    std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << path << std::endl;

    //// 打印前10个点的信息
    //for (size_t i = 0; i < 10 && i < cloud->points.size(); ++i)
    //{
    //    std::cout << " x : " << cloud->points[i].x
    //        << " y : " << cloud->points[i].y
    //        << " z : " << cloud->points[i].z << std::endl;
    //}

    // 创建 KdTree 对象
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // 创建法向量估计对象
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);

    // 设置 K 近邻数
    int k = 10;// 使用 k 个最近邻点
    int K = 25;// 使用 K 个最近邻点
    ne.setKSearch(k);

    // 计算法向量
    ne.compute(*normals);

    // 打印前10个点的法线信息
    for (size_t i = 0; i < 10 && i < normals->points.size(); ++i)
    {
        std::cout << "Normal for point " << i << ": "
            << "nx: " << normals->points[i].normal_x
            << " ny: " << normals->points[i].normal_y
            << " nz: " << normals->points[i].normal_z
            << " curvature: " << normals->points[i].curvature << std::endl;
    }




    ////////循环step1
    std::vector<float> weighted_centroid_projection_metric;//加权质心投影项
    std::vector<float> centroid_projection_term;//质心投影项
    Eigen::MatrixXf centroids(3, cloud->points.size());// 创建一个矩阵来存储每个点的 K 近邻质心
    std::vector<float> normal_angle_stddevs; // 存储每个点的法线夹角标准差
    std::vector<float> dot_product_stddevs; // 存储每个点的k临近点最小二乘距离标准差
    float normal_angle_stddevs_weight = 500;//法线夹角标准差权重参数
    float dot_product_stddevs_weight = 100;//k临近点最小二乘距离标准差权重参数    
    std::vector<std::vector<int>> adjacency_matrix(cloud->points.size(), std::vector<int>(K, -1));// 创建一个 int 型矩阵来存储邻接点序号

    

    // 遍历每个点，计算其k近邻的相关数据
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {      
        std::vector<int> indices;//用于存储第i个点的K近邻点序号
        std::vector<float> distances;//用于存储第i个点的K近邻点距离
        if (tree->nearestKSearch(cloud->points[i], K, indices, distances) > 0)//查询K近邻点
        {
            //计算K近邻点质心
            std::vector<pcl::PointXYZ> neighbors;//用于存储K近邻点坐标
            for (int j = 0; j < K; ++j)
            {
                neighbors.push_back(cloud->points[indices[j]]);//K近邻点坐标存储              
                adjacency_matrix[i][j] = indices[j]; // 填充邻接点序号矩阵
            }
            Eigen::Vector3f centroid = computeCentroid(neighbors);// 利用computeCentroid函数计算质心
            centroids.col(i) = centroid;//计算质心并存到centroids
            
            
                                        
            //计算质心投影项
            Eigen::Vector3f diff = Eigen::Vector3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z) - centroid;// 计算点与质心的差值
            float dot_product = std::abs(diff.dot(Eigen::Vector3f(normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z)));// 计算差值与法向量的点积并取绝对值
            centroid_projection_term.push_back(dot_product);// 存储结果



            //计算法线夹角的标准差
            std::vector<float> angles;//用于存储法向夹角（弧度）数组
            for (int j = 0; j < K; ++j)
            {
                Eigen::Vector3f neighbor_normal(normals->points[indices[j]].normal_x, normals->points[indices[j]].normal_y, normals->points[indices[j]].normal_z);//k邻近点法向
                float angle = computeAngleBetweenNormals(Eigen::Vector3f(normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z), neighbor_normal);//i,j法向夹角（角度）
                angles.push_back(angle);//存储法向夹角（角度）
                //printf("angle%d:%f\n", j, angle);
            }
            float stddev = computeStandardDeviation(angles)/ normal_angle_stddevs_weight;// 法向夹角标准差*权重参数
            normal_angle_stddevs.push_back(stddev);// 存储法向夹角标准差*权重参数



            //计算每个点的K邻近点最小二乘距离标准差
            std::vector<float> dot_products; // 用于存储点积绝对值
            for (int j = 0; j < K; ++j)
            {
                Eigen::Vector3f neighbor_point(cloud->points[indices[j]].x, cloud->points[indices[j]].y, cloud->points[indices[j]].z);//K邻近点坐标
                Eigen::Vector3f diff_neighbor = Eigen::Vector3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z) - neighbor_point;//K邻近点方向向量
                float dot_product_neighbor = std::abs(diff_neighbor.dot(Eigen::Vector3f(normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z)));//K邻近点最小二乘距离
                dot_products.push_back(dot_product_neighbor);//存储K临近点最小二乘距离
            }
            float dot_product_stddev = computeStandardDeviation(dot_products)/ dot_product_stddevs_weight;//K邻近点最小二乘距离标准差*权重参数
            dot_product_stddevs.push_back(dot_product_stddev);//存储K邻近点最小二乘距离标准差*权重参数




            // 计算加权质心投影项
            float weighted_projection = centroid_projection_term.back() * std::exp(normal_angle_stddevs.back()) * std::exp(dot_product_stddevs.back());// 计算加权质心投影项
            weighted_centroid_projection_metric.push_back(weighted_projection);// 存储加权质心投影项
            
        }       
    }

    //for (size_t i = 0; i < weighted_centroid_projection_metric.size(); ++i)//打印加权质心投影项
    //{
    //    std::cout << "weighted_centroid_projection_metric" << i << ": "
    //    << weighted_centroid_projection_metric[i] << std::endl;
    //}
    //for (size_t i = 0; i < normal_angle_stddevs.size(); ++i)//打印加权质心投影项
    //{
    //    std::cout << "normal_angle_stddevs" << i << ": "
    //    << normal_angle_stddevs[i] << std::endl;
    //}

    // 排序加权质心投影项
    std::vector<std::pair<size_t, float>> indexed_metrics;//创建点对
    for (size_t i = 0; i < weighted_centroid_projection_metric.size(); ++i)
    {
        indexed_metrics.emplace_back(i, weighted_centroid_projection_metric[i]);//初始化点对
    }
    std::sort(indexed_metrics.begin(), indexed_metrics.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
        return a.second < b.second;//排序点对
        });

    //for (size_t i = 0; i < indexed_metrics.size(); ++i)//打印排序点对
    //{
    //    std::cout << "indexed_metrics" << i << ": "
    //        << "first" << indexed_metrics[i].first
    //        << "second" << indexed_metrics[i].second << std::endl;
    //}

    // 计算前alpha%和后beta%的序号
    float alpha = 0.15;
    float beta = 0.15;
    size_t total_points = indexed_metrics.size();//总数
    size_t num_top = static_cast<size_t>(total_points * alpha);//前alpha%数
    size_t num_bottom = static_cast<size_t>(total_points * beta);//后beta%数
    std::vector<size_t> top_indices;//用于存储前alpha%序号
    std::vector<size_t> bottom_indices;//用于存储后beta%序号
    std::vector<size_t> middle_indices;//用于存储中间序号
    for (size_t i = 0; i < num_top; ++i)
    {
        top_indices.push_back(indexed_metrics[total_points - 1 - i].first);//存储前alpha%序号
    }
    for (size_t i = 0; i < num_bottom; ++i)
    {
        bottom_indices.push_back(indexed_metrics[i].first);//用于存储后beta%序号
    }
    for (size_t i = num_bottom; i < total_points - num_top; ++i)
    {
        middle_indices.push_back(indexed_metrics[i].first);//用于存储中间序号
    }



    // 输出序号
    //std::cout << "Top 20% indices: ";
    //for (const auto& idx : top_indices)
    //{
    //    std::cout << idx << " ";
    //}
    //std::cout << std::endl;
    //std::cout << "Bottom 15% indices: ";
    //for (const auto& idx : bottom_indices)
    //{
    //    std::cout << idx << " ";
    //}
    //std::cout << std::endl;
    //std::cout << "Middle 65% indices: ";
    //for (const auto& idx : middle_indices)
    //{
    //    std::cout << idx << " ";
    //}
    //std::cout << std::endl;



    // 传递中间的数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_step2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_step2(new pcl::PointCloud<pcl::Normal>);
    Eigen::MatrixXf centroids_step2(3, middle_indices.size());
    std::vector<float> normal_angle_stddevs_step2;
    std::vector<float> dot_product_stddevs_step2;
    size_t loop_count = 0; // 引入一个计数器来跟踪循环次数
    for (const auto& idx : middle_indices)
    {
        cloud_step2->push_back(cloud->points[idx]);//传递点云
        normals_step2->push_back(normals->points[idx]);//传递法向
        centroids_step2.col(loop_count) = centroids.col(idx);// 传递每个点的 K 近邻质心
        normal_angle_stddevs_step2.push_back(normal_angle_stddevs[idx]);//传递法线夹角的标准差
        dot_product_stddevs_step2.push_back(dot_product_stddevs[idx]);//传递k临近点最小二乘距离标准差
        ++loop_count; // 增加循环计数器
    }

    //printf("%d      %d", adjacency_matrix[6036][5], adjacency_matrix[middle_indices[0]][5]);

    // 创建可视化对象
    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    viewer.setBackgroundColor(0, 0, 0); // 设置背景颜色为黑色
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud"); // 添加点云
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud"); // 设置点的大小
    viewer.initCameraParameters(); // 初始化相机参数


    // 将前alpha%的点变为红色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color_handler(cloud, 255, 0, 0);
    pcl::PointCloud<pcl::PointXYZ>::Ptr top_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : top_indices)
    {
        top_cloud->push_back(cloud->points[idx]);
    }
    viewer.addPointCloud<pcl::PointXYZ>(top_cloud, red_color_handler, "top cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "top cloud");


    
    ////////循环step2
    std::vector<std::vector<int>> adjacency_matrix_step2(cloud_step2->points.size(), std::vector<int>(K, -1));// 创建一个 int 型矩阵来存储邻接点序号
    std::vector<float> centroid_projection_term_step2(cloud_step2->points.size(), 0);//质心投影项
    std::vector<float> weighted_centroid_projection_metric_step2;//加权质心投影项
    Eigen::MatrixXf centroids_step2_to_step3(3, cloud_step2->points.size());// 创建一个矩阵来存储每个点的新近邻质心

    //建立循环
    for (size_t i = 0; i < cloud_step2->points.size(); ++i)
    {
        Eigen::Vector3f point_i(cloud_step2->points[i].x, cloud_step2->points[i].y, cloud_step2->points[i].z);//第i个点坐标
        for (size_t j = 0; j < adjacency_matrix[middle_indices[i]].size(); ++j)
        {
            
            

            //计算第i个点与质心与其K个近邻点张成平面的法向量
            Eigen::Vector3f point_i_j(cloud->points[adjacency_matrix[middle_indices[i]][j]].x, cloud->points[adjacency_matrix[middle_indices[i]][j]].y, cloud->points[adjacency_matrix[middle_indices[i]][j]].z);//第i个点的第j个近邻点坐标
            Eigen::Vector3f centroid = centroids_step2.col(i);//第i个点K个近邻点的质心
            Eigen::Vector3f vector1 = centroid - point_i;//质心-第i个点
            Eigen::Vector3f vector2 = point_i - point_i_j;//第i个点-第i个点的第j个近邻点坐标
            Eigen::Vector3f cross_product = vector1.cross(vector2);//v1，v2叉乘
            float cross_product_norm = cross_product.norm();//cross_product的模
            std::vector<int> neighbors_1; // 存储点积结果大于等于0的近邻点序号
            std::vector<int> neighbors_2; // 存储点积结果小于等于0的近邻点序号
            float centroid_projection_term_max = 0; //质心投影项最大值           
            if (cross_product_norm > 0)
            {


            
                //计算点积结果大于等于0以及点积结果小于等于0的近邻点序号
                for (size_t m = 0; m < adjacency_matrix[middle_indices[i]].size(); ++m)
                {
                    Eigen::Vector3f point_i_m(cloud->points[adjacency_matrix[middle_indices[i]][m]].x, cloud->points[adjacency_matrix[middle_indices[i]][m]].y, cloud->points[adjacency_matrix[middle_indices[i]][m]].z);//第i个点的第m个近邻点坐标
                    Eigen::Vector3f diff = point_i - point_i_m; // 第i个点与其K近邻点的差
                    float dot_product = diff.dot(cross_product); // 与cross_product作点积
                    if (dot_product >= 0)
                    {
                        neighbors_1.push_back(adjacency_matrix[middle_indices[i]][m]); // 点积结果大于等于0的近邻点序号
                    }
                    if (dot_product <= 0)
                    {
                        neighbors_2.push_back(adjacency_matrix[middle_indices[i]][m]); // 点积结果小于等于0的近邻点序号
                    }
                }
                

                
                //计算点积结果大于等于0的质心投影项并更新近邻点与质心
                std::vector<pcl::PointXYZ> neighbors_xyz_1;// 存储点积结果大于等于0的近邻点坐标
                for (int m = 0; m < neighbors_1.size(); ++m)
                {
                    neighbors_xyz_1.push_back(cloud->points[neighbors_1[m]]);//近邻点坐标存储
                }
                Eigen::Vector3f centroid_1 = computeCentroid(neighbors_xyz_1);// 利用computeCentroid函数计算质心
                Eigen::Vector3f diff = point_i - centroid_1;// 计算点与质心的差值
                float dot_product = std::abs(diff.dot(Eigen::Vector3f(normals_step2->points[i].normal_x, normals_step2->points[i].normal_y, normals_step2->points[i].normal_z)));// 计算差值与法向量的点积并取绝对值
                if (dot_product > centroid_projection_term_max)
                {
                    centroid_projection_term_max = dot_product;//更新最大值
                    centroid_projection_term_step2[i] = centroid_projection_term_max;// 存储最大质心投影项
                    centroids_step2_to_step3.col(i) = centroid_1;//存储质心到centroids_step2_to_step3
                    for (int m = 0; m < neighbors_1.size(); ++m)
                    {
                        adjacency_matrix_step2[i][m] = neighbors_1[m];//更新近邻点
                    }
                    for (int m = neighbors_1.size(); m < adjacency_matrix_step2[i].size(); ++m)
                    {
                        adjacency_matrix_step2[i][m] = -1;//更新近邻点，多余部分设置为-1
                    }                    
                }


                
                //计算点积结果小于等于0的质心投影项并更新近邻点与质心
                std::vector<pcl::PointXYZ> neighbors_xyz_2;// 存储点积结果小于等于0的近邻点坐标
                for (int m = 0; m < neighbors_2.size(); ++m)
                {
                    neighbors_xyz_2.push_back(cloud->points[neighbors_2[m]]);//近邻点坐标存储
                }
                Eigen::Vector3f centroid_2 = computeCentroid(neighbors_xyz_2);// 利用computeCentroid函数计算质心
                diff = point_i - centroid_2;// 计算点与质心的差值
                dot_product = std::abs(diff.dot(Eigen::Vector3f(normals_step2->points[i].normal_x, normals_step2->points[i].normal_y, normals_step2->points[i].normal_z)));// 计算差值与法向量的点积并取绝对值
                if (dot_product > centroid_projection_term_max)
                {
                    centroid_projection_term_max = dot_product;//更新最大值
                    centroid_projection_term_step2[i] = centroid_projection_term_max;// 存储最大质心投影项
                    centroids_step2_to_step3.col(i) = centroid_2;//存储质心到centroids_step2_to_step3
                    for (int m = 0; m < neighbors_2.size(); ++m)
                    {
                        adjacency_matrix_step2[i][m] = neighbors_2[m];//更新近邻点
                    }
                    for (int m = neighbors_2.size(); m < adjacency_matrix_step2[i].size(); ++m)
                    {
                        adjacency_matrix_step2[i][m] = -1;//更新近邻点，多余部分设置为-1
                    }
                }
            }
        }  
        
        
        
        // 计算加权质心投影项
        float weighted_projection = centroid_projection_term_step2[i] * std::exp(normal_angle_stddevs_step2[i]) * std::exp(dot_product_stddevs_step2[i]);// 计算加权质心投影项
        weighted_centroid_projection_metric_step2.push_back(weighted_projection);// 存储加权质心投影项
    }
                                   
    //for (size_t i = 0; i < weighted_centroid_projection_metric_step2.size(); ++i)
    //{
    //    std::cout << "weighted_centroid_projection_metric_step2" << i << ": "
    //    << weighted_centroid_projection_metric_step2[i] << std::endl;
    //}

    // 排序加权质心投影项
    std::vector<std::pair<size_t, float>> indexed_metrics_step2;//创建点对
    for (size_t i = 0; i < weighted_centroid_projection_metric_step2.size(); ++i)
    {
        indexed_metrics_step2.emplace_back(i, weighted_centroid_projection_metric_step2[i]);//初始化点对
    }
    std::sort(indexed_metrics_step2.begin(), indexed_metrics_step2.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
        return a.second < b.second;//排序点对
        });

    //for (size_t i = 0; i < weighted_centroid_projection_metric_step2.size(); ++i)
    //{
    //    std::cout << "indexed_metrics_step" << i << ": "
    //        <<"first" << indexed_metrics_step2[i].first 
    //        << "second" << indexed_metrics_step2[i].second<<std::endl;
    //}

    // 计算前alpha%和后beta%的序号
    float alpha_step2 = 0.075;
    float beta_step2 = 0.15;
    size_t total_points_step2 = indexed_metrics_step2.size();//总数
    size_t num_top_step2 = static_cast<size_t>(total_points_step2 * alpha_step2);//前alpha%数
    size_t num_bottom_step2 = static_cast<size_t>(total_points_step2 * beta_step2);//后beta%数
    std::vector<size_t> top_indices_step2;//用于存储前alpha%序号
    std::vector<size_t> bottom_indices_step2;//用于存储后beta%序号
    std::vector<size_t> middle_indices_step2;//用于存储中间序号
    for (size_t i = 0; i < num_top_step2; ++i)
    {
        top_indices_step2.push_back(indexed_metrics_step2[total_points_step2 - 1 - i].first);//存储前alpha%序号
    }
    for (size_t i = 0; i < num_bottom_step2; ++i)
    {
        bottom_indices_step2.push_back(indexed_metrics_step2[i].first);//用于存储后beta%序号
    }
    for (size_t i = num_bottom_step2; i < total_points_step2 - num_top_step2; ++i)
    {
        middle_indices_step2.push_back(indexed_metrics_step2[i].first);//用于存储中间序号
    }
    
    //for (size_t i = 0; i < top_indices_step2.size(); ++i)
    //{
    //    std::cout << "top_indices_step2" << i << ": "
    //        << top_indices_step2[i] << std::endl;
    //}

    // 传递中间的数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_step3(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_step3(new pcl::PointCloud<pcl::Normal>);
    Eigen::MatrixXf centroids_step3(3, middle_indices_step2.size());
    std::vector<float> normal_angle_stddevs_step3;
    std::vector<float> dot_product_stddevs_step3;
    size_t loop_count2 = 0; // 引入一个计数器来跟踪循环次数
    for (const auto& idx : middle_indices_step2)
    {
        cloud_step3->push_back(cloud_step2->points[idx]);//传递点云
        normals_step3->push_back(normals_step2->points[idx]);//传递法向
        centroids_step3.col(loop_count2) = centroids_step2_to_step3.col(idx);// 传递每个点的 K 近邻质心
        normal_angle_stddevs_step3.push_back(normal_angle_stddevs_step2[idx]);//传递法线夹角的标准差
        dot_product_stddevs_step3.push_back(dot_product_stddevs_step2[idx]);//传递k临近点最小二乘距离标准差
        ++loop_count2; // 增加循环计数器
    }



    // 将前alpha%的点变为红色
    pcl::PointCloud<pcl::PointXYZ>::Ptr top_cloud_step2(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : top_indices_step2)
    {
        top_cloud_step2->push_back(cloud_step2->points[idx]);
    }
    viewer.addPointCloud<pcl::PointXYZ>(top_cloud_step2, red_color_handler, "top cloud step2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "top cloud step2");

    

    ////////循环step3
    std::vector<float> centroid_projection_term_step3(cloud_step2->points.size(), 0);//质心投影项
    std::vector<float> weighted_centroid_projection_metric_step3;//加权质心投影项

    //建立循环
    for (size_t i = 0; i < cloud_step3->points.size(); ++i)
    {
        Eigen::Vector3f point_i(cloud_step3->points[i].x, cloud_step3->points[i].y, cloud_step3->points[i].z);//第i个点坐标
        
        for (size_t j = 0; j < adjacency_matrix_step2[middle_indices_step2[i]].size(); ++j)
        {
            if (adjacency_matrix_step2[middle_indices_step2[i]][j] >= 0)
            {
                //计算第i个点与质心与其K个近邻点张成平面的法向量
                Eigen::Vector3f point_i_j(cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][j]].x, cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][j]].y, cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][j]].z);//第i个点的第j个近邻点坐标
                Eigen::Vector3f centroid = centroids_step3.col(i);//第i个点K个近邻点的质心
                Eigen::Vector3f vector1 = centroid - point_i;//质心-第i个点
                Eigen::Vector3f vector2 = point_i - point_i_j;//第i个点-第i个点的第j个近邻点坐标
                Eigen::Vector3f cross_product = vector1.cross(vector2);//v1，v2叉乘
                float cross_product_norm = cross_product.norm();//cross_product的模
                std::vector<int> neighbors_1_step3; // 存储点积结果大于等于0的近邻点序号
                std::vector<int> neighbors_2_step3; // 存储点积结果小于等于0的近邻点序号
                float centroid_projection_term_max = 0; //质心投影项最大值           
                if (cross_product_norm > 0)
                {



                    //计算点积结果大于等于0以及点积结果小于等于0的近邻点序号
                    for (size_t m = 0; m < adjacency_matrix_step2[middle_indices_step2[i]].size() ; ++m)
                    {
                        if (adjacency_matrix_step2[middle_indices_step2[i]][m] >= 0)
                        {
                            Eigen::Vector3f point_i_m(cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][m]].x, cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][m]].y, cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][m]].z);//第i个点的第m个近邻点坐标
                            Eigen::Vector3f diff = point_i - point_i_m; // 第i个点与其K近邻点的差
                            float dot_product = diff.dot(cross_product); // 与cross_product作点积
                            if (dot_product >= 0)
                            {
                                neighbors_1_step3.push_back(adjacency_matrix_step2[middle_indices_step2[i]][m]); // 点积结果大于等于0的近邻点序号
                            }
                            if (dot_product <= 0)
                            {
                                neighbors_2_step3.push_back(adjacency_matrix_step2[middle_indices_step2[i]][m]); // 点积结果大于等于0的近邻点序号
                            }
                        }
                    }



                    //计算点积结果大于等于0的质心投影项并更新近邻点与质心
                    std::vector<pcl::PointXYZ> neighbors_xyz_1;// 存储点积结果大于等于0的近邻点坐标
                    for (int m = 0; m < neighbors_1_step3.size(); ++m)
                    {
                        neighbors_xyz_1.push_back(cloud->points[neighbors_1_step3[m]]);//近邻点坐标存储
                    }
                    Eigen::Vector3f centroid_1 = computeCentroid(neighbors_xyz_1);// 利用computeCentroid函数计算质心
                    Eigen::Vector3f diff = point_i - centroid_1;// 计算点与质心的差值
                    float dot_product = std::abs(diff.dot(Eigen::Vector3f(normals_step3->points[i].normal_x, normals_step3->points[i].normal_y, normals_step3->points[i].normal_z)));// 计算差值与法向量的点积并取绝对值
                    if (dot_product > centroid_projection_term_max)
                    {
                        centroid_projection_term_max = dot_product;//更新最大值
                        centroid_projection_term_step3[i] = centroid_projection_term_max;// 存储最大质心投影项
                    }



                    //计算点积结果小于等于0的质心投影项并更新近邻点与质心
                    std::vector<pcl::PointXYZ> neighbors_xyz_2;// 存储点积结果小于等于0的近邻点坐标
                    for (int m = 0; m < neighbors_2_step3.size(); ++m)
                    {
                        neighbors_xyz_2.push_back(cloud->points[neighbors_2_step3[m]]);//近邻点坐标存储
                    }
                    Eigen::Vector3f centroid_2 = computeCentroid(neighbors_xyz_2);// 利用computeCentroid函数计算质心
                    diff = point_i - centroid_2;// 计算点与质心的差值
                    dot_product = std::abs(diff.dot(Eigen::Vector3f(normals_step3->points[i].normal_x, normals_step3->points[i].normal_y, normals_step3->points[i].normal_z)));// 计算差值与法向量的点积并取绝对值
                    if (dot_product > centroid_projection_term_max)
                    {
                        centroid_projection_term_max = dot_product;//更新最大值
                        centroid_projection_term_step3[i] = centroid_projection_term_max;// 存储最大质心投影项
                        centroids_step2_to_step3.col(i) = centroid_2;//存储质心到centroids_step2_to_step3
                    }
                }
            

            
            }
        }

        

        // 计算加权质心投影项
        float weighted_projection = centroid_projection_term_step3[i] * std::exp(normal_angle_stddevs_step3[i]) * std::exp(dot_product_stddevs_step3[i]);// 计算加权质心投影项
        weighted_centroid_projection_metric_step3.push_back(weighted_projection);// 存储加权质心投影项
    }

    

    // 排序加权质心投影项
    std::vector<std::pair<size_t, float>> indexed_metrics_step3;//创建点对
    for (size_t i = 0; i < weighted_centroid_projection_metric_step3.size(); ++i)
    {
        indexed_metrics_step3.emplace_back(i, weighted_centroid_projection_metric_step3[i]);//初始化点对
    }
    std::sort(indexed_metrics_step3.begin(), indexed_metrics_step3.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
        return a.second < b.second;//排序点对
        });



    // 计算前alpha%和后beta%的序号
    float alpha_step3 = 0.075;
    float beta_step3 = 0.15;
    size_t total_points_step3 = indexed_metrics_step3.size();//总数
    size_t num_top_step3 = static_cast<size_t>(total_points_step3 * alpha_step3);//前alpha%数
    size_t num_bottom_step3 = static_cast<size_t>(total_points_step3 * beta_step3);//后beta%数
    std::vector<size_t> top_indices_step3;//用于存储前alpha%序号
    std::vector<size_t> bottom_indices_step3;//用于存储后beta%序号
    std::vector<size_t> middle_indices_step3;//用于存储中间序号
    for (size_t i = 0; i < num_top_step3; ++i)
    {
        top_indices_step3.push_back(indexed_metrics_step3[total_points_step3 - 1 - i].first);//存储前alpha%序号
    }
    for (size_t i = 0; i < num_bottom_step3; ++i)
    {
        bottom_indices_step3.push_back(indexed_metrics_step3[i].first);//用于存储后beta%序号
    }
    for (size_t i = num_bottom_step3; i < total_points_step3 - num_top_step3; ++i)
    {
        middle_indices_step3.push_back(indexed_metrics_step3[i].first);//用于存储中间序号
    }



    // 将前alpha%的点变为红色
    pcl::PointCloud<pcl::PointXYZ>::Ptr top_cloud_step3(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : top_indices_step3)
    {
        top_cloud_step3->push_back(cloud_step3->points[idx]);
    }
    viewer.addPointCloud<pcl::PointXYZ>(top_cloud_step3, red_color_handler, "top cloud step3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "top cloud step3");

    // 可视化法向量
    //float max_line_length = 0.01; // 法线的长度
    //viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 20, max_line_length);//第三个参数表示间隔


    // 启动可视化窗口
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;

}