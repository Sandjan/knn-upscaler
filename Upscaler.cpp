#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <utility>

class Upscaler
{
private:
    int diam;
    int padding;
    cv::Mat image;
    int n;
    int topk;
    int stop;
    int scale;
    int middle;
    int areaSize;
    cv::Mat downscaled;
    double colorPreservation;
    std::vector<int> context_weights;

public:
    // Constructor
    Upscaler(std::string imgPath, int diam, int topk, std::vector<int> &context_weights, int areaSize, int scale, int stop, double colorPreservation)
    {
        this->image = cv::imread(imgPath, cv::IMREAD_COLOR);
        if (this->image.empty())
        {
            throw "Konnte das Bild nicht öffnen.";
        }
        cv::resize(this->image, this->downscaled, cv::Size(), 1.0 / scale, 1.0 / scale, cv::INTER_AREA);
        this->scale = scale;
        this->topk = topk;
        this->areaSize = areaSize / scale;
        this->diam = diam;
        this->middle = diam * diam / 2;
        this->padding = int(diam / 2);
        this->context_weights = std::move(context_weights);
        this->stop = stop;
        this->colorPreservation = colorPreservation;
    }

    int calcDist(int y, int x, int j, int i)
    {
        int dist = 0;
        for (int yy = -padding; yy <= padding; yy++) // rotierungen beachten
        {
            for (int xx = -padding; xx <= padding; xx++)
            {
                // Loop over each color channel
                for (int c = 0; c < 3; c++)
                {
                    uint8_t p1 = downscaled.at<cv::Vec3b>(j + yy, i + xx)[c];
                    uint8_t p2 = image.at<cv::Vec3b>(y + yy, x + xx)[c];
                    int d = (p1 > p2) ? (p1 - p2) : (p2 - p1);
                    int d_y = (yy > 0) ? yy : -yy;
                    int d_x = (xx > 0) ? xx : -xx;
                    dist += d * context_weights[(d_y > d_x) ? d_y : d_x];
                }
            }
        }
        return dist;
    }

    void findBestFits(int y, int x, std::pair<int, int> *min_indices)
    {
        int min_dists[topk];
        std::fill_n(min_dists, topk, 99999);
        int high_dist_i = 0;
        int top = y / scale - areaSize;
        int bottom = y / scale + areaSize;
        int left = x / scale - areaSize;
        int right = x / scale + areaSize;

        if (top < padding)
        {
            top = padding;
        }
        if (bottom > downscaled.rows - padding)
        {
            bottom = downscaled.rows - padding;
        }
        if (left < padding)
        {
            left = padding;
        }
        if (right > downscaled.cols - padding)
        {
            right = downscaled.cols - padding;
        }

        for (int j = top; j < bottom; ++j)
        {
            for (int i = left; i < right; ++i)
            {
                int d = 0;
                for (int c = 0; c < 3; c++)
                {
                    uint8_t p1 = downscaled.at<cv::Vec3b>(j, i)[c];
                    uint8_t p2 = image.at<cv::Vec3b>(y, x)[c];
                    d += (p1 > p2) ? (p1 - p2) : (p2 - p1);
                }
                if (d < stop) // skip if middle pixel is too different
                {
                    int dist = calcDist(y, x, j, i);
                    if (dist < min_dists[high_dist_i]) // wenn die aktuelle distanz besser ist als die schlechteste
                    {
                        min_dists[high_dist_i] = dist; // setzte diese an die stelle des schlechtesten
                        min_indices[high_dist_i].first = j;
                        min_indices[high_dist_i].second = i;
                        for (int i = 0; i < topk; ++i) // neue schlechteste distanz bestimmen
                        {
                            if (min_dists[i] > min_dists[high_dist_i])
                            {                    // wenn die aktuelle distanz der topk größer ist als die aktuell schlechteste
                                high_dist_i = i; // setzte den index für die aktuell schlechteste auf den aktuellen
                            }
                        }
                    }
                }
            }
        }
    }

    void fillPadding(cv::Mat &upscaled)
    {
        for (int y = 0; y < image.rows; y++)
        {
            int y_hr = y * scale;
            for (int x = 0; x < image.cols; x++)
            {
                if (x >= padding && x < image.cols - padding && y >= padding && y < image.rows - padding)
                {
                    continue;
                }
                int x_hr = x * scale;
                for (int c = 0; c < 3; c++)
                {
                    for (int i = 0; i < scale * scale; ++i) // set all pixels
                    {
                        upscaled.at<cv::Vec3b>(y_hr + i / scale, x_hr + i % scale)[c] = image.at<cv::Vec3b>(y, x)[c];
                    }
                }
            }
        }
    }

    cv::Mat
    calculateUpscaled()
    {
        cv::Mat upscaled = cv::Mat::zeros(image.rows * scale, image.cols * scale, image.type());

#pragma omp parallel for
        for (int y = padding; y < image.rows - padding; y++)
        {
            int y_hr = y * scale;
            if (y % 5 == 0)
            {
                std::cout << double(y) / image.rows << "%" << std::endl;
            }
            for (int x = padding; x < image.cols - padding; x++)
            {
                int x_hr = x * scale;
                std::pair<int, int> bestFits_y_x[topk];
                std::fill_n(bestFits_y_x, topk, std::make_pair(-1, -1));
                findBestFits(y, x, bestFits_y_x);
                for (int c = 0; c < 3; c++)
                {
                    std::vector<std::vector<double>> vals(scale, std::vector<double>(scale));
                    double sumc = 0.0;
                    for (int yy = 0; yy < scale; ++yy) // set all pixels
                    {
                        for (int xx = 0; xx < scale; ++xx)
                        {
                            int num = 0;
                            double sum = 0;
                            for (int j = 0; j < topk; ++j) // calculate average of topk
                            {
                                if (bestFits_y_x[j].first != -1)
                                {
                                    sum += image.at<cv::Vec3b>(bestFits_y_x[j].first * scale + yy, bestFits_y_x[j].second * scale + xx)[c];
                                    num++;
                                }
                            }
                            if (num == 0)
                            {
                                vals[yy][xx] = image.at<cv::Vec3b>(y, x)[c];
                            }
                            else
                            {
                                vals[yy][xx] = sum / num;
                            }
                            sumc += vals[yy][xx];
                        }
                    }
                    double corrector = (double(image.at<cv::Vec3b>(y, x)[c]) * (scale * scale)) / sumc; // make color preservation after upscaling shifted one pixel right and down for reducing pixelation
                    corrector = (corrector * colorPreservation + (1.0 - colorPreservation));
                    for (int yy = 0; yy < scale; ++yy) // set all pixels
                    {
                        for (int xx = 0; xx < scale; ++xx)
                        {
                            int ne = vals[yy][xx] * corrector + 0.5;
                            upscaled.at<cv::Vec3b>(y_hr + yy, x_hr + xx)[c] = ne > 255 ? 255 : ne;
                        }
                    }
                }
            }
        }
        // fillPadding(upscaled);
        return upscaled;
    }
};

int main(int argc, char **argv)
{
    std::string image_path;
    std::string output_path = "upscaled.png";
    int context_diam = 5;
    int topK = 1;
    int areaSize = 30;
    int scale = 2;
    double colorPreservation = 0.0;
    int stop = 50;
    std::vector<int> context_weights;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-i")
        {
            image_path = argv[++i];
        }
        else if (arg == "-o")
        {
            output_path = argv[++i];
        }
        else if (arg == "-c")
        {
            context_diam = std::stoi(argv[++i]);
        }
        else if (arg == "--topK")
        {
            topK = std::stoi(argv[++i]);
        }
        else if (arg == "--context-weights")
        {
            std::string input = argv[++i];
            std::stringstream ss(input);
            std::string token;
            while (std::getline(ss, token, ';'))
            {
                context_weights.push_back(std::stoi(token));
            }
        }
        else if (arg == "--area-size")
        {
            areaSize = std::stoi(argv[++i]);
        }
        else if (arg == "--scale")
        {
            scale = std::stoi(argv[++i]);
        }
        else if (arg == "--colorPreservation")
        {
            colorPreservation = std::stod(argv[++i]);
        }
        else if (arg == "--stop")
        {
            stop = std::stoi(argv[++i]);
        }
    }

    std::cout << "Processing with params:\n"
              << "  context diameter:  " << context_diam
              << "\n  averaging the top: " << topK << std::endl;

    std::cout << "  Context weights (distance from middle):\n";
    for (int i = 0; i <= context_diam / 2; ++i)
    {
        std::cout << "    distance=" << i << "; w=" << context_weights[i] << std::endl;
        int temp = ((i * 2 + 1) - 1) * 4;
        if (temp == 0)
        {
            temp++;
        }
        int multip = ((context_diam - 1) * 4) / temp;
        context_weights[i] *= multip; // multiplier to adjust weights to context pixel count
    }
    std::cout << "\nCalculating...\n";

    Upscaler up(image_path, context_diam, topK, context_weights, areaSize, scale, stop, colorPreservation);

    cv::imwrite(output_path, up.calculateUpscaled());

    return 0;
}
