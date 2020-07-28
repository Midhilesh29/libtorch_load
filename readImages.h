#include <sys/types.h>
#include <torch/torch.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>

typedef std::vector<std::string> stringvec;
using Data = std::vector<std::pair<torch::Tensor, torch::Tensor>>;
 
void read_directory(const std::string& name, stringvec& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;

    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(dp->d_name);
    }

    closedir(dirp);
}

std::pair<std::vector<stringvec>, std::vector<int>> read_images(std::string files_path) {
    stringvec classes{"gatto", "mucca", "pecora", "scoiattolo"};
    stringvec::iterator  dir_name, image_name;
    std::vector<stringvec> images_path;
    std::vector<int> labels;
    const char *dot = "."; //directories to  be discarded
    const char *dot1 = "..";//directories to  be discarded
    int dir_count = 0;

    for (dir_name = classes.begin(); dir_name != classes.end(); dir_name++) {
        stringvec images;
        stringvec dir_images;
        read_directory(files_path + *dir_name, images);

        for (image_name = images.begin();  image_name != images.end(); image_name++) {
            std::string name = *image_name;

            if (name.compare(dot) and name.compare(dot1)) {
                dir_images.push_back(files_path + *dir_name + "/" + name);
                labels.push_back(dir_count);
            }
        }

        images_path.push_back(dir_images);
        dir_count++;
    }

    return std::make_pair(images_path, labels);
}

Data convertData(std::pair<std::vector<stringvec>, std::vector<int>> data_path, bool train) {
    stringvec::iterator img_name;
    std::vector<int>::iterator l;

    std::vector<stringvec> images_path = data_path.first;
    std::vector<int> labels = data_path.second;
    std::string index;

    Data final_tensorData;


    for (int i = 0; i < images_path.size(); i++) {
        int total_images_count  = images_path[i].size();
        int train_images_count = 0.85*total_images_count;
        int start, end;

        // std::cout << train_images_count << " " + std::to_string(total_images_count) << std::endl;

        if (train) {
            start = 0;
            end = train_images_count;
            index = "train";
        } else  {
            start = train_images_count;
            end = total_images_count;
            index = "test";
        }

        
        for (img_name = images_path[i].begin() + start; img_name != images_path[i].begin() + end; img_name++) {
            cv::Mat image = cv::imread(*img_name);
            // We convert opencv BGR to RGB as pretrained models trained on RGB.
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::Size scale(224, 224);
            //Resize the image to standard size for classification
            cv::resize(image, image, scale, 0, 0, cv::INTER_AREA);
            
            // Convert Mat image to tensor 1 x H x W x C
            torch::Tensor tensorImage = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, at::kByte);
            tensorImage = tensorImage.to(torch::kFloat32).div_(255);

            tensorImage = at::transpose(tensorImage, 0, 1);
            tensorImage = at::transpose(tensorImage, 0, 2);

            torch::Tensor tLabel = torch::tensor(i);
            final_tensorData.push_back(std::make_pair(tensorImage, tLabel));        
        }
    }

    if (train) {
        std::random_shuffle(final_tensorData.begin(), final_tensorData.end());
    }

    return final_tensorData;
}
