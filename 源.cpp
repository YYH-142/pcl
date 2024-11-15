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

// ����һ��������
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

// ���㷨��֮��ļнǣ����ȣ�
float computeAngleBetweenNormals(const Eigen::Vector3f& normal1, const Eigen::Vector3f& normal2)
{
    if (normal1.dot(normal2) / (normal1.norm() * normal2.norm()) > 1)
    {
        return 0;
    }
    return std::acos(normal1.dot(normal2) / (normal1.norm() * normal2.norm()));
}

// �����׼��
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
    // ����������������
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // ��ȡPCD�ļ�
    const char* path = "D:\\pcd_model\\fandisk-normdirectionnoise0.25.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file %s\n", path);
        return -1;
    }
    std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << path << std::endl;

    //// ��ӡǰ10�������Ϣ
    //for (size_t i = 0; i < 10 && i < cloud->points.size(); ++i)
    //{
    //    std::cout << " x : " << cloud->points[i].x
    //        << " y : " << cloud->points[i].y
    //        << " z : " << cloud->points[i].z << std::endl;
    //}

    // ���� KdTree ����
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // �������������ƶ���
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);

    // ���� K ������
    int k = 10;// ʹ�� k ������ڵ�
    int K = 25;// ʹ�� K ������ڵ�
    ne.setKSearch(k);

    // ���㷨����
    ne.compute(*normals);

    // ��ӡǰ10����ķ�����Ϣ
    for (size_t i = 0; i < 10 && i < normals->points.size(); ++i)
    {
        std::cout << "Normal for point " << i << ": "
            << "nx: " << normals->points[i].normal_x
            << " ny: " << normals->points[i].normal_y
            << " nz: " << normals->points[i].normal_z
            << " curvature: " << normals->points[i].curvature << std::endl;
    }




    ////////ѭ��step1
    std::vector<float> weighted_centroid_projection_metric;//��Ȩ����ͶӰ��
    std::vector<float> centroid_projection_term;//����ͶӰ��
    Eigen::MatrixXf centroids(3, cloud->points.size());// ����һ���������洢ÿ����� K ��������
    std::vector<float> normal_angle_stddevs; // �洢ÿ����ķ��߼нǱ�׼��
    std::vector<float> dot_product_stddevs; // �洢ÿ�����k�ٽ�����С���˾����׼��
    float normal_angle_stddevs_weight = 500;//���߼нǱ�׼��Ȩ�ز���
    float dot_product_stddevs_weight = 100;//k�ٽ�����С���˾����׼��Ȩ�ز���    
    std::vector<std::vector<int>> adjacency_matrix(cloud->points.size(), std::vector<int>(K, -1));// ����һ�� int �;������洢�ڽӵ����

    

    // ����ÿ���㣬������k���ڵ��������
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {      
        std::vector<int> indices;//���ڴ洢��i�����K���ڵ����
        std::vector<float> distances;//���ڴ洢��i�����K���ڵ����
        if (tree->nearestKSearch(cloud->points[i], K, indices, distances) > 0)//��ѯK���ڵ�
        {
            //����K���ڵ�����
            std::vector<pcl::PointXYZ> neighbors;//���ڴ洢K���ڵ�����
            for (int j = 0; j < K; ++j)
            {
                neighbors.push_back(cloud->points[indices[j]]);//K���ڵ�����洢              
                adjacency_matrix[i][j] = indices[j]; // ����ڽӵ���ž���
            }
            Eigen::Vector3f centroid = computeCentroid(neighbors);// ����computeCentroid������������
            centroids.col(i) = centroid;//�������Ĳ��浽centroids
            
            
                                        
            //��������ͶӰ��
            Eigen::Vector3f diff = Eigen::Vector3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z) - centroid;// ����������ĵĲ�ֵ
            float dot_product = std::abs(diff.dot(Eigen::Vector3f(normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z)));// �����ֵ�뷨�����ĵ����ȡ����ֵ
            centroid_projection_term.push_back(dot_product);// �洢���



            //���㷨�߼нǵı�׼��
            std::vector<float> angles;//���ڴ洢����нǣ����ȣ�����
            for (int j = 0; j < K; ++j)
            {
                Eigen::Vector3f neighbor_normal(normals->points[indices[j]].normal_x, normals->points[indices[j]].normal_y, normals->points[indices[j]].normal_z);//k�ڽ��㷨��
                float angle = computeAngleBetweenNormals(Eigen::Vector3f(normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z), neighbor_normal);//i,j����нǣ��Ƕȣ�
                angles.push_back(angle);//�洢����нǣ��Ƕȣ�
                //printf("angle%d:%f\n", j, angle);
            }
            float stddev = computeStandardDeviation(angles)/ normal_angle_stddevs_weight;// ����нǱ�׼��*Ȩ�ز���
            normal_angle_stddevs.push_back(stddev);// �洢����нǱ�׼��*Ȩ�ز���



            //����ÿ�����K�ڽ�����С���˾����׼��
            std::vector<float> dot_products; // ���ڴ洢�������ֵ
            for (int j = 0; j < K; ++j)
            {
                Eigen::Vector3f neighbor_point(cloud->points[indices[j]].x, cloud->points[indices[j]].y, cloud->points[indices[j]].z);//K�ڽ�������
                Eigen::Vector3f diff_neighbor = Eigen::Vector3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z) - neighbor_point;//K�ڽ��㷽������
                float dot_product_neighbor = std::abs(diff_neighbor.dot(Eigen::Vector3f(normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z)));//K�ڽ�����С���˾���
                dot_products.push_back(dot_product_neighbor);//�洢K�ٽ�����С���˾���
            }
            float dot_product_stddev = computeStandardDeviation(dot_products)/ dot_product_stddevs_weight;//K�ڽ�����С���˾����׼��*Ȩ�ز���
            dot_product_stddevs.push_back(dot_product_stddev);//�洢K�ڽ�����С���˾����׼��*Ȩ�ز���




            // �����Ȩ����ͶӰ��
            float weighted_projection = centroid_projection_term.back() * std::exp(normal_angle_stddevs.back()) * std::exp(dot_product_stddevs.back());// �����Ȩ����ͶӰ��
            weighted_centroid_projection_metric.push_back(weighted_projection);// �洢��Ȩ����ͶӰ��
            
        }       
    }

    //for (size_t i = 0; i < weighted_centroid_projection_metric.size(); ++i)//��ӡ��Ȩ����ͶӰ��
    //{
    //    std::cout << "weighted_centroid_projection_metric" << i << ": "
    //    << weighted_centroid_projection_metric[i] << std::endl;
    //}
    //for (size_t i = 0; i < normal_angle_stddevs.size(); ++i)//��ӡ��Ȩ����ͶӰ��
    //{
    //    std::cout << "normal_angle_stddevs" << i << ": "
    //    << normal_angle_stddevs[i] << std::endl;
    //}

    // �����Ȩ����ͶӰ��
    std::vector<std::pair<size_t, float>> indexed_metrics;//�������
    for (size_t i = 0; i < weighted_centroid_projection_metric.size(); ++i)
    {
        indexed_metrics.emplace_back(i, weighted_centroid_projection_metric[i]);//��ʼ�����
    }
    std::sort(indexed_metrics.begin(), indexed_metrics.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
        return a.second < b.second;//������
        });

    //for (size_t i = 0; i < indexed_metrics.size(); ++i)//��ӡ������
    //{
    //    std::cout << "indexed_metrics" << i << ": "
    //        << "first" << indexed_metrics[i].first
    //        << "second" << indexed_metrics[i].second << std::endl;
    //}

    // ����ǰalpha%�ͺ�beta%�����
    float alpha = 0.15;
    float beta = 0.15;
    size_t total_points = indexed_metrics.size();//����
    size_t num_top = static_cast<size_t>(total_points * alpha);//ǰalpha%��
    size_t num_bottom = static_cast<size_t>(total_points * beta);//��beta%��
    std::vector<size_t> top_indices;//���ڴ洢ǰalpha%���
    std::vector<size_t> bottom_indices;//���ڴ洢��beta%���
    std::vector<size_t> middle_indices;//���ڴ洢�м����
    for (size_t i = 0; i < num_top; ++i)
    {
        top_indices.push_back(indexed_metrics[total_points - 1 - i].first);//�洢ǰalpha%���
    }
    for (size_t i = 0; i < num_bottom; ++i)
    {
        bottom_indices.push_back(indexed_metrics[i].first);//���ڴ洢��beta%���
    }
    for (size_t i = num_bottom; i < total_points - num_top; ++i)
    {
        middle_indices.push_back(indexed_metrics[i].first);//���ڴ洢�м����
    }



    // ������
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



    // �����м������
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_step2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_step2(new pcl::PointCloud<pcl::Normal>);
    Eigen::MatrixXf centroids_step2(3, middle_indices.size());
    std::vector<float> normal_angle_stddevs_step2;
    std::vector<float> dot_product_stddevs_step2;
    size_t loop_count = 0; // ����һ��������������ѭ������
    for (const auto& idx : middle_indices)
    {
        cloud_step2->push_back(cloud->points[idx]);//���ݵ���
        normals_step2->push_back(normals->points[idx]);//���ݷ���
        centroids_step2.col(loop_count) = centroids.col(idx);// ����ÿ����� K ��������
        normal_angle_stddevs_step2.push_back(normal_angle_stddevs[idx]);//���ݷ��߼нǵı�׼��
        dot_product_stddevs_step2.push_back(dot_product_stddevs[idx]);//����k�ٽ�����С���˾����׼��
        ++loop_count; // ����ѭ��������
    }

    //printf("%d      %d", adjacency_matrix[6036][5], adjacency_matrix[middle_indices[0]][5]);

    // �������ӻ�����
    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    viewer.setBackgroundColor(0, 0, 0); // ���ñ�����ɫΪ��ɫ
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud"); // ��ӵ���
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud"); // ���õ�Ĵ�С
    viewer.initCameraParameters(); // ��ʼ���������


    // ��ǰalpha%�ĵ��Ϊ��ɫ
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color_handler(cloud, 255, 0, 0);
    pcl::PointCloud<pcl::PointXYZ>::Ptr top_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : top_indices)
    {
        top_cloud->push_back(cloud->points[idx]);
    }
    viewer.addPointCloud<pcl::PointXYZ>(top_cloud, red_color_handler, "top cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "top cloud");


    
    ////////ѭ��step2
    std::vector<std::vector<int>> adjacency_matrix_step2(cloud_step2->points.size(), std::vector<int>(K, -1));// ����һ�� int �;������洢�ڽӵ����
    std::vector<float> centroid_projection_term_step2(cloud_step2->points.size(), 0);//����ͶӰ��
    std::vector<float> weighted_centroid_projection_metric_step2;//��Ȩ����ͶӰ��
    Eigen::MatrixXf centroids_step2_to_step3(3, cloud_step2->points.size());// ����һ���������洢ÿ������½�������

    //����ѭ��
    for (size_t i = 0; i < cloud_step2->points.size(); ++i)
    {
        Eigen::Vector3f point_i(cloud_step2->points[i].x, cloud_step2->points[i].y, cloud_step2->points[i].z);//��i��������
        for (size_t j = 0; j < adjacency_matrix[middle_indices[i]].size(); ++j)
        {
            
            

            //�����i��������������K�����ڵ��ų�ƽ��ķ�����
            Eigen::Vector3f point_i_j(cloud->points[adjacency_matrix[middle_indices[i]][j]].x, cloud->points[adjacency_matrix[middle_indices[i]][j]].y, cloud->points[adjacency_matrix[middle_indices[i]][j]].z);//��i����ĵ�j�����ڵ�����
            Eigen::Vector3f centroid = centroids_step2.col(i);//��i����K�����ڵ������
            Eigen::Vector3f vector1 = centroid - point_i;//����-��i����
            Eigen::Vector3f vector2 = point_i - point_i_j;//��i����-��i����ĵ�j�����ڵ�����
            Eigen::Vector3f cross_product = vector1.cross(vector2);//v1��v2���
            float cross_product_norm = cross_product.norm();//cross_product��ģ
            std::vector<int> neighbors_1; // �洢���������ڵ���0�Ľ��ڵ����
            std::vector<int> neighbors_2; // �洢������С�ڵ���0�Ľ��ڵ����
            float centroid_projection_term_max = 0; //����ͶӰ�����ֵ           
            if (cross_product_norm > 0)
            {


            
                //������������ڵ���0�Լ�������С�ڵ���0�Ľ��ڵ����
                for (size_t m = 0; m < adjacency_matrix[middle_indices[i]].size(); ++m)
                {
                    Eigen::Vector3f point_i_m(cloud->points[adjacency_matrix[middle_indices[i]][m]].x, cloud->points[adjacency_matrix[middle_indices[i]][m]].y, cloud->points[adjacency_matrix[middle_indices[i]][m]].z);//��i����ĵ�m�����ڵ�����
                    Eigen::Vector3f diff = point_i - point_i_m; // ��i��������K���ڵ�Ĳ�
                    float dot_product = diff.dot(cross_product); // ��cross_product�����
                    if (dot_product >= 0)
                    {
                        neighbors_1.push_back(adjacency_matrix[middle_indices[i]][m]); // ���������ڵ���0�Ľ��ڵ����
                    }
                    if (dot_product <= 0)
                    {
                        neighbors_2.push_back(adjacency_matrix[middle_indices[i]][m]); // ������С�ڵ���0�Ľ��ڵ����
                    }
                }
                

                
                //������������ڵ���0������ͶӰ����½��ڵ�������
                std::vector<pcl::PointXYZ> neighbors_xyz_1;// �洢���������ڵ���0�Ľ��ڵ�����
                for (int m = 0; m < neighbors_1.size(); ++m)
                {
                    neighbors_xyz_1.push_back(cloud->points[neighbors_1[m]]);//���ڵ�����洢
                }
                Eigen::Vector3f centroid_1 = computeCentroid(neighbors_xyz_1);// ����computeCentroid������������
                Eigen::Vector3f diff = point_i - centroid_1;// ����������ĵĲ�ֵ
                float dot_product = std::abs(diff.dot(Eigen::Vector3f(normals_step2->points[i].normal_x, normals_step2->points[i].normal_y, normals_step2->points[i].normal_z)));// �����ֵ�뷨�����ĵ����ȡ����ֵ
                if (dot_product > centroid_projection_term_max)
                {
                    centroid_projection_term_max = dot_product;//�������ֵ
                    centroid_projection_term_step2[i] = centroid_projection_term_max;// �洢�������ͶӰ��
                    centroids_step2_to_step3.col(i) = centroid_1;//�洢���ĵ�centroids_step2_to_step3
                    for (int m = 0; m < neighbors_1.size(); ++m)
                    {
                        adjacency_matrix_step2[i][m] = neighbors_1[m];//���½��ڵ�
                    }
                    for (int m = neighbors_1.size(); m < adjacency_matrix_step2[i].size(); ++m)
                    {
                        adjacency_matrix_step2[i][m] = -1;//���½��ڵ㣬���ಿ������Ϊ-1
                    }                    
                }


                
                //���������С�ڵ���0������ͶӰ����½��ڵ�������
                std::vector<pcl::PointXYZ> neighbors_xyz_2;// �洢������С�ڵ���0�Ľ��ڵ�����
                for (int m = 0; m < neighbors_2.size(); ++m)
                {
                    neighbors_xyz_2.push_back(cloud->points[neighbors_2[m]]);//���ڵ�����洢
                }
                Eigen::Vector3f centroid_2 = computeCentroid(neighbors_xyz_2);// ����computeCentroid������������
                diff = point_i - centroid_2;// ����������ĵĲ�ֵ
                dot_product = std::abs(diff.dot(Eigen::Vector3f(normals_step2->points[i].normal_x, normals_step2->points[i].normal_y, normals_step2->points[i].normal_z)));// �����ֵ�뷨�����ĵ����ȡ����ֵ
                if (dot_product > centroid_projection_term_max)
                {
                    centroid_projection_term_max = dot_product;//�������ֵ
                    centroid_projection_term_step2[i] = centroid_projection_term_max;// �洢�������ͶӰ��
                    centroids_step2_to_step3.col(i) = centroid_2;//�洢���ĵ�centroids_step2_to_step3
                    for (int m = 0; m < neighbors_2.size(); ++m)
                    {
                        adjacency_matrix_step2[i][m] = neighbors_2[m];//���½��ڵ�
                    }
                    for (int m = neighbors_2.size(); m < adjacency_matrix_step2[i].size(); ++m)
                    {
                        adjacency_matrix_step2[i][m] = -1;//���½��ڵ㣬���ಿ������Ϊ-1
                    }
                }
            }
        }  
        
        
        
        // �����Ȩ����ͶӰ��
        float weighted_projection = centroid_projection_term_step2[i] * std::exp(normal_angle_stddevs_step2[i]) * std::exp(dot_product_stddevs_step2[i]);// �����Ȩ����ͶӰ��
        weighted_centroid_projection_metric_step2.push_back(weighted_projection);// �洢��Ȩ����ͶӰ��
    }
                                   
    //for (size_t i = 0; i < weighted_centroid_projection_metric_step2.size(); ++i)
    //{
    //    std::cout << "weighted_centroid_projection_metric_step2" << i << ": "
    //    << weighted_centroid_projection_metric_step2[i] << std::endl;
    //}

    // �����Ȩ����ͶӰ��
    std::vector<std::pair<size_t, float>> indexed_metrics_step2;//�������
    for (size_t i = 0; i < weighted_centroid_projection_metric_step2.size(); ++i)
    {
        indexed_metrics_step2.emplace_back(i, weighted_centroid_projection_metric_step2[i]);//��ʼ�����
    }
    std::sort(indexed_metrics_step2.begin(), indexed_metrics_step2.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
        return a.second < b.second;//������
        });

    //for (size_t i = 0; i < weighted_centroid_projection_metric_step2.size(); ++i)
    //{
    //    std::cout << "indexed_metrics_step" << i << ": "
    //        <<"first" << indexed_metrics_step2[i].first 
    //        << "second" << indexed_metrics_step2[i].second<<std::endl;
    //}

    // ����ǰalpha%�ͺ�beta%�����
    float alpha_step2 = 0.075;
    float beta_step2 = 0.15;
    size_t total_points_step2 = indexed_metrics_step2.size();//����
    size_t num_top_step2 = static_cast<size_t>(total_points_step2 * alpha_step2);//ǰalpha%��
    size_t num_bottom_step2 = static_cast<size_t>(total_points_step2 * beta_step2);//��beta%��
    std::vector<size_t> top_indices_step2;//���ڴ洢ǰalpha%���
    std::vector<size_t> bottom_indices_step2;//���ڴ洢��beta%���
    std::vector<size_t> middle_indices_step2;//���ڴ洢�м����
    for (size_t i = 0; i < num_top_step2; ++i)
    {
        top_indices_step2.push_back(indexed_metrics_step2[total_points_step2 - 1 - i].first);//�洢ǰalpha%���
    }
    for (size_t i = 0; i < num_bottom_step2; ++i)
    {
        bottom_indices_step2.push_back(indexed_metrics_step2[i].first);//���ڴ洢��beta%���
    }
    for (size_t i = num_bottom_step2; i < total_points_step2 - num_top_step2; ++i)
    {
        middle_indices_step2.push_back(indexed_metrics_step2[i].first);//���ڴ洢�м����
    }
    
    //for (size_t i = 0; i < top_indices_step2.size(); ++i)
    //{
    //    std::cout << "top_indices_step2" << i << ": "
    //        << top_indices_step2[i] << std::endl;
    //}

    // �����м������
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_step3(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_step3(new pcl::PointCloud<pcl::Normal>);
    Eigen::MatrixXf centroids_step3(3, middle_indices_step2.size());
    std::vector<float> normal_angle_stddevs_step3;
    std::vector<float> dot_product_stddevs_step3;
    size_t loop_count2 = 0; // ����һ��������������ѭ������
    for (const auto& idx : middle_indices_step2)
    {
        cloud_step3->push_back(cloud_step2->points[idx]);//���ݵ���
        normals_step3->push_back(normals_step2->points[idx]);//���ݷ���
        centroids_step3.col(loop_count2) = centroids_step2_to_step3.col(idx);// ����ÿ����� K ��������
        normal_angle_stddevs_step3.push_back(normal_angle_stddevs_step2[idx]);//���ݷ��߼нǵı�׼��
        dot_product_stddevs_step3.push_back(dot_product_stddevs_step2[idx]);//����k�ٽ�����С���˾����׼��
        ++loop_count2; // ����ѭ��������
    }



    // ��ǰalpha%�ĵ��Ϊ��ɫ
    pcl::PointCloud<pcl::PointXYZ>::Ptr top_cloud_step2(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : top_indices_step2)
    {
        top_cloud_step2->push_back(cloud_step2->points[idx]);
    }
    viewer.addPointCloud<pcl::PointXYZ>(top_cloud_step2, red_color_handler, "top cloud step2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "top cloud step2");

    

    ////////ѭ��step3
    std::vector<float> centroid_projection_term_step3(cloud_step2->points.size(), 0);//����ͶӰ��
    std::vector<float> weighted_centroid_projection_metric_step3;//��Ȩ����ͶӰ��

    //����ѭ��
    for (size_t i = 0; i < cloud_step3->points.size(); ++i)
    {
        Eigen::Vector3f point_i(cloud_step3->points[i].x, cloud_step3->points[i].y, cloud_step3->points[i].z);//��i��������
        
        for (size_t j = 0; j < adjacency_matrix_step2[middle_indices_step2[i]].size(); ++j)
        {
            if (adjacency_matrix_step2[middle_indices_step2[i]][j] >= 0)
            {
                //�����i��������������K�����ڵ��ų�ƽ��ķ�����
                Eigen::Vector3f point_i_j(cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][j]].x, cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][j]].y, cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][j]].z);//��i����ĵ�j�����ڵ�����
                Eigen::Vector3f centroid = centroids_step3.col(i);//��i����K�����ڵ������
                Eigen::Vector3f vector1 = centroid - point_i;//����-��i����
                Eigen::Vector3f vector2 = point_i - point_i_j;//��i����-��i����ĵ�j�����ڵ�����
                Eigen::Vector3f cross_product = vector1.cross(vector2);//v1��v2���
                float cross_product_norm = cross_product.norm();//cross_product��ģ
                std::vector<int> neighbors_1_step3; // �洢���������ڵ���0�Ľ��ڵ����
                std::vector<int> neighbors_2_step3; // �洢������С�ڵ���0�Ľ��ڵ����
                float centroid_projection_term_max = 0; //����ͶӰ�����ֵ           
                if (cross_product_norm > 0)
                {



                    //������������ڵ���0�Լ�������С�ڵ���0�Ľ��ڵ����
                    for (size_t m = 0; m < adjacency_matrix_step2[middle_indices_step2[i]].size() ; ++m)
                    {
                        if (adjacency_matrix_step2[middle_indices_step2[i]][m] >= 0)
                        {
                            Eigen::Vector3f point_i_m(cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][m]].x, cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][m]].y, cloud->points[adjacency_matrix_step2[middle_indices_step2[i]][m]].z);//��i����ĵ�m�����ڵ�����
                            Eigen::Vector3f diff = point_i - point_i_m; // ��i��������K���ڵ�Ĳ�
                            float dot_product = diff.dot(cross_product); // ��cross_product�����
                            if (dot_product >= 0)
                            {
                                neighbors_1_step3.push_back(adjacency_matrix_step2[middle_indices_step2[i]][m]); // ���������ڵ���0�Ľ��ڵ����
                            }
                            if (dot_product <= 0)
                            {
                                neighbors_2_step3.push_back(adjacency_matrix_step2[middle_indices_step2[i]][m]); // ���������ڵ���0�Ľ��ڵ����
                            }
                        }
                    }



                    //������������ڵ���0������ͶӰ����½��ڵ�������
                    std::vector<pcl::PointXYZ> neighbors_xyz_1;// �洢���������ڵ���0�Ľ��ڵ�����
                    for (int m = 0; m < neighbors_1_step3.size(); ++m)
                    {
                        neighbors_xyz_1.push_back(cloud->points[neighbors_1_step3[m]]);//���ڵ�����洢
                    }
                    Eigen::Vector3f centroid_1 = computeCentroid(neighbors_xyz_1);// ����computeCentroid������������
                    Eigen::Vector3f diff = point_i - centroid_1;// ����������ĵĲ�ֵ
                    float dot_product = std::abs(diff.dot(Eigen::Vector3f(normals_step3->points[i].normal_x, normals_step3->points[i].normal_y, normals_step3->points[i].normal_z)));// �����ֵ�뷨�����ĵ����ȡ����ֵ
                    if (dot_product > centroid_projection_term_max)
                    {
                        centroid_projection_term_max = dot_product;//�������ֵ
                        centroid_projection_term_step3[i] = centroid_projection_term_max;// �洢�������ͶӰ��
                    }



                    //���������С�ڵ���0������ͶӰ����½��ڵ�������
                    std::vector<pcl::PointXYZ> neighbors_xyz_2;// �洢������С�ڵ���0�Ľ��ڵ�����
                    for (int m = 0; m < neighbors_2_step3.size(); ++m)
                    {
                        neighbors_xyz_2.push_back(cloud->points[neighbors_2_step3[m]]);//���ڵ�����洢
                    }
                    Eigen::Vector3f centroid_2 = computeCentroid(neighbors_xyz_2);// ����computeCentroid������������
                    diff = point_i - centroid_2;// ����������ĵĲ�ֵ
                    dot_product = std::abs(diff.dot(Eigen::Vector3f(normals_step3->points[i].normal_x, normals_step3->points[i].normal_y, normals_step3->points[i].normal_z)));// �����ֵ�뷨�����ĵ����ȡ����ֵ
                    if (dot_product > centroid_projection_term_max)
                    {
                        centroid_projection_term_max = dot_product;//�������ֵ
                        centroid_projection_term_step3[i] = centroid_projection_term_max;// �洢�������ͶӰ��
                        centroids_step2_to_step3.col(i) = centroid_2;//�洢���ĵ�centroids_step2_to_step3
                    }
                }
            

            
            }
        }

        

        // �����Ȩ����ͶӰ��
        float weighted_projection = centroid_projection_term_step3[i] * std::exp(normal_angle_stddevs_step3[i]) * std::exp(dot_product_stddevs_step3[i]);// �����Ȩ����ͶӰ��
        weighted_centroid_projection_metric_step3.push_back(weighted_projection);// �洢��Ȩ����ͶӰ��
    }

    

    // �����Ȩ����ͶӰ��
    std::vector<std::pair<size_t, float>> indexed_metrics_step3;//�������
    for (size_t i = 0; i < weighted_centroid_projection_metric_step3.size(); ++i)
    {
        indexed_metrics_step3.emplace_back(i, weighted_centroid_projection_metric_step3[i]);//��ʼ�����
    }
    std::sort(indexed_metrics_step3.begin(), indexed_metrics_step3.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
        return a.second < b.second;//������
        });



    // ����ǰalpha%�ͺ�beta%�����
    float alpha_step3 = 0.075;
    float beta_step3 = 0.15;
    size_t total_points_step3 = indexed_metrics_step3.size();//����
    size_t num_top_step3 = static_cast<size_t>(total_points_step3 * alpha_step3);//ǰalpha%��
    size_t num_bottom_step3 = static_cast<size_t>(total_points_step3 * beta_step3);//��beta%��
    std::vector<size_t> top_indices_step3;//���ڴ洢ǰalpha%���
    std::vector<size_t> bottom_indices_step3;//���ڴ洢��beta%���
    std::vector<size_t> middle_indices_step3;//���ڴ洢�м����
    for (size_t i = 0; i < num_top_step3; ++i)
    {
        top_indices_step3.push_back(indexed_metrics_step3[total_points_step3 - 1 - i].first);//�洢ǰalpha%���
    }
    for (size_t i = 0; i < num_bottom_step3; ++i)
    {
        bottom_indices_step3.push_back(indexed_metrics_step3[i].first);//���ڴ洢��beta%���
    }
    for (size_t i = num_bottom_step3; i < total_points_step3 - num_top_step3; ++i)
    {
        middle_indices_step3.push_back(indexed_metrics_step3[i].first);//���ڴ洢�м����
    }



    // ��ǰalpha%�ĵ��Ϊ��ɫ
    pcl::PointCloud<pcl::PointXYZ>::Ptr top_cloud_step3(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : top_indices_step3)
    {
        top_cloud_step3->push_back(cloud_step3->points[idx]);
    }
    viewer.addPointCloud<pcl::PointXYZ>(top_cloud_step3, red_color_handler, "top cloud step3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "top cloud step3");

    // ���ӻ�������
    //float max_line_length = 0.01; // ���ߵĳ���
    //viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 20, max_line_length);//������������ʾ���


    // �������ӻ�����
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;

}