#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#define PI 3.14159265

// hero status struct
struct HeroStatus {
    int hero_id;
    cv::Point position;
    int level;
};

// all kinds of status within one frame are stored in this struct
struct FrameStatus {
    int ts;    // timestamp unit: ms
    double joystick_angle;    // angle between joystick direction and horizontal, scaled [-180, 180)
    int spell1_cd;
    int spell2_cd;
    int spell3_cd;
    int skill1_cd;
    int skill2_cd;
    int skill3_cd;
    int skill4_cd;
    int money;
    std::vector<HeroStatus> hero_list;
};

class GameVideoAnalyzer {
  private:   
    // fixed locations
    double radius_spell_;
    double radius_skill_;
    double joystick_height_;
    double joystick_width_;
    cv::Point joystick_lu_;
    cv::Point joystick_axis_;

    // list of distance between joystick and its axis
    std::vector<double> dist_list_;

    // current unoccupied hero id
    int hero_id_;

    bool is_heroes_list_initialized_;

    //list of last updated timestamp for heroes in heroes_list_
    std::vector<int> last_updated_;

    //list of times of appearance for heroes in heroes_list_
    std::vector<int> appearances_;
    
  public:
    // list of status per frame
    std::vector<FrameStatus> status_list_;

    // list of heroes appeared in the sequence
    std::vector<HeroStatus> heroes_list_;

    GameVideoAnalyzer();
    void adjust_size(cv::Mat*);
    int detect_number_roi(cv::Mat*, const cv::Rect&, const std::vector<cv::Mat>&, const double&);
    int detect_number_fixed(cv::Mat*, const std::vector<cv::Mat>&, const double&, const size_t&, const cv::Vec4d&);
    double estimate_joystick_angle(cv::Mat*);
    void estimate_js_axis_status(double*, double*);
    void track_hero(cv::Mat*, std::vector<HeroStatus>*, const int& ts, const std::vector<cv::Mat>&, const cv::Mat&, const double&, const size_t&);
    bool is_black_white(const cv::Mat&) const;
    void assign_hero(const int&, const cv::Point&, std::vector<HeroStatus>*, const int&);
    void delete_inactive_heroes(const int&, const int&, const int&);
    inline void update_frame_status(const FrameStatus& frame_status) {
        status_list_.push_back(frame_status);
    }
};

GameVideoAnalyzer::GameVideoAnalyzer() {
    // joystick locations are hardcoded
    joystick_height_ = 309.0;
    joystick_width_ = 294.0;
    joystick_lu_.x = 58;
    joystick_lu_.y = 411;
    joystick_axis_.x = 206;
    joystick_axis_.y = 559;
    // cv::Point joystick_axis_(201, 568);    // var = 8.2252
    // cv::Point joystick_axis_(206, 559);    // var = 7.7941
    // cv::Point joystick_axis_(196, 569);    // var = 9.3208

    // set size and capacity of vectors as 0
    std::vector<FrameStatus>().swap(status_list_);
    std::vector<double>().swap(dist_list_);
    std::vector<int>().swap(last_updated_);
    std::vector<int>().swap(appearances_);
    std::vector<HeroStatus>().swap(heroes_list_);

    // set hero id as 0
    hero_id_ = 0;

    is_heroes_list_initialized_ = false;
}

void GameVideoAnalyzer::adjust_size(cv::Mat* frame) {
    // always resize frame image to 1280*720
    if (frame->cols != 1280 || frame->rows != 720) {
        cv::resize(*frame, *frame, cv::Size(1280, 720), 0, 0, cv::INTER_LINEAR);
    }
}

int GameVideoAnalyzer::detect_number_roi(cv::Mat* src, const cv::Rect& box, const std::vector<cv::Mat>& number_samples, const double& avg_err_thres) {
    double min_avg_err = avg_err_thres;   // averge error threshold
    int number_detected = -1;

    for (size_t i = 0; i < 10; i++) {
        // std::cout << i << ',';
        cv::Mat src_roi = (*src)(box);
        cv::Mat number_sample = number_samples[i];

        // check h/w ratio
        double hw_ratio_roi = static_cast<double>(src_roi.rows) / static_cast<double>(src_roi.cols);
        double hw_ratio_sample = static_cast<double>(number_sample.rows) / static_cast<double>(number_sample.cols);
        if (hw_ratio_roi / hw_ratio_sample > 1.2 || hw_ratio_roi / hw_ratio_sample < 0.8) {
            continue;
        }

        cv::resize(src_roi, src_roi, cv::Size(number_sample.cols, number_sample.rows), 0, 0, cv::INTER_NEAREST);

        // for visualization
        cv::Mat number_compare(src_roi.rows, src_roi.cols * 3, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat number_diff = number_compare(cv::Rect(src_roi.cols * 2, 0, src_roi.cols, src_roi.rows));
        cv::cvtColor(src_roi, number_compare(cv::Rect(0, 0, src_roi.cols, src_roi.rows)), cv::COLOR_GRAY2BGR);
        cv::cvtColor(number_sample, number_compare(cv::Rect(src_roi.cols, 0, src_roi.cols, src_roi.rows)), cv::COLOR_GRAY2BGR);

        double avg_err = 0;
        for (size_t iy = 0; iy < src_roi.rows; iy++) {
            for (size_t ix = 0; ix < src_roi.cols; ix++) {
                if (src_roi.at<uchar>(iy, ix) == 0xff && number_sample.at<uchar>(iy, ix) == 0) {
                    number_diff.at<cv::Vec3b>(iy, ix) = cv::Vec3b(0, 0, 255);
                    avg_err += 1.0f;
                } else if (src_roi.at<uchar>(iy, ix) == 0 && number_sample.at<uchar>(iy, ix) == 0xff) {
                    number_diff.at<cv::Vec3b>(iy, ix) = cv::Vec3b(0, 255, 0);
                    avg_err += 1.0f;
                } else if (src_roi.at<uchar>(iy, ix) == number_sample.at<uchar>(iy, ix)) {
                    number_diff.at<cv::Vec3b>(iy, ix) = cv::Vec3b(255, 0, 0);
                }
            }
        }
        avg_err /= static_cast<double>(src_roi.cols * src_roi.rows);
        // std::cout << avg_err << std::endl;
        if (avg_err < min_avg_err) {
            min_avg_err = avg_err;
            number_detected = i;
        }

        cv::namedWindow("cur, sam, dif");
        cv::imshow("cur, sam, dif", number_compare);
        // cv::waitKey(0);
    }
    return number_detected;
}

int GameVideoAnalyzer::detect_number_fixed(cv::Mat* src, const std::vector<cv::Mat>& number_samples, const double& avg_err_thres, const size_t& bw_thres, const cv::Vec4d& size_restrict) {
    // pass cropped number image into this function
    cv::cvtColor(*src, *src, cv::COLOR_BGR2GRAY);
    cv::threshold(*src, *src, bw_thres, 255, cv::THRESH_BINARY);

    cv::namedWindow("src_bw");
    cv::imshow("src_bw", *src);

    std::vector<std::vector<cv::Point>> contours;
    std::map<int, int> coord_num_map;   // x coordinate and number detected
    cv::findContours(*src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect number_box = cv::boundingRect(contours[i]);
        // std::cout << number_box << std::endl;

        // size of segmented regions are restricted
        // according to a ratio to the size if input is all 0
        if (size_restrict == cv::Vec4d(0, 0, 0, 0)) {
            if (static_cast<double>(number_box.height) > static_cast<double>(src->rows) / 1.3 ||
                static_cast<double>(number_box.height) < static_cast<double>(src->rows) / 1.6) {
                continue;
            }
            if (static_cast<double>(number_box.width) > static_cast<double>(src->cols) / 4.1 ||
                static_cast<double>(number_box.width) < static_cast<double>(src->cols) / 9.3) {
                continue;
            }
        } else {
            // or according to input restrictions
            // size_restrict = cv::Vec4b(height_min, height_max, width_min, width_max)
            if (number_box.height > size_restrict[1] || number_box.height < size_restrict[0]) {
                continue;
            }
            if (number_box.width > size_restrict[3] || number_box.width < size_restrict[2]) {
                continue;
            }
        }

        int number_detected = GameVideoAnalyzer::detect_number_roi(src, number_box, number_samples, avg_err_thres);
        if (number_detected != -1) {
            coord_num_map.insert(std::pair<int, int>(number_box.x, number_detected));
        }
    }

    int cooldown = 0;
    for (std::map<int, int>::iterator it = coord_num_map.begin(); it != coord_num_map.end(); it++) {
        cooldown *= 10;
        cooldown += it->second;
    }
    return cooldown;
}

double GameVideoAnalyzer::estimate_joystick_angle(cv::Mat* src) {
    cv::Mat joystick_rect = (*src)(cv::Rect(joystick_lu_.x, joystick_lu_.y, joystick_width_, joystick_height_));
    cv::Mat joystick_gray;
    cv::cvtColor(joystick_rect, joystick_gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(joystick_gray, circles, cv::HOUGH_GRADIENT, 1, 100, 50, 20, 40, 50);
    cv::line(*src, (joystick_axis_ - cv::Point(10, 0)), (joystick_axis_ + cv::Point(30, 0)), cv::Scalar(0, 0, 255), 3);
    cv::line(*src, (joystick_axis_ - cv::Point(0, 10)), (joystick_axis_ + cv::Point(0, 10)), cv::Scalar(0, 0, 255), 3);
    double joystick_angle = 666.0;
    if (!circles.empty()) {
        cv::Point circle_center = joystick_lu_ + cv::Point(circles[0][0], circles[0][1]);
        cv::circle(*src, circle_center, circles[0][2], cv::Scalar(0, 0, 255), 3);
        cv::line(*src, joystick_axis_, circle_center, cv::Scalar(0, 0, 255), 3);
        dist_list_.push_back(sqrt(pow(circle_center.x - joystick_axis_.x, 2) + pow(circle_center.y - joystick_axis_.y, 2)));
        joystick_angle = atan2(joystick_axis_.y - circle_center.y, circle_center.x - joystick_axis_.x) * 180 / PI;
    }
    return joystick_angle;
}

void GameVideoAnalyzer::estimate_js_axis_status(double* mean, double* stdvar) {
    // std var is used to validate joystick axis coordinates
    if (dist_list_.empty()) {
        return;
    }
    *mean = std::accumulate(std::begin(dist_list_), std::end(dist_list_), 0) / static_cast<double>(dist_list_.size());
    double err_sum = 0;
    for (size_t i = 0; i < dist_list_.size(); i++) {
        err_sum += pow(dist_list_[i] - *mean, 2);
    }
    *stdvar = sqrt(err_sum / (dist_list_.size() - 1));
}

bool GameVideoAnalyzer::is_black_white(const cv::Mat& src) const {
    int color_pixels = 0;
    cv::Mat src_hsv;
    std::vector<cv::Mat> src_hsv_vec;
    cv::cvtColor(src, src_hsv, cv::COLOR_BGR2HSV);
    cv::split(src_hsv, src_hsv_vec);
    for (size_t iy = 0; iy < src.rows; iy++) {
        for (size_t ix = 0; ix < src.cols; ix++) {
            // check Saturation and Value in HSV space
            if (src_hsv_vec[1].at<uchar>(iy, ix) > 70 && src_hsv_vec[2].at<uchar>(iy, ix) > 30) {
                // std::cout << "color pixel with rgb = " << src.at<cv::Vec3b>(iy, ix) << " and hsv = " << src_hsv.at<cv::Vec3b>(iy, ix) << std::endl;
                color_pixels++;
            }
        }
    }
    std::cout << "Color pixels " << color_pixels;
    if (color_pixels < std::max(12, src.cols * 2)) {
        std::cout << " ROI bw check succeed!" << std::endl;
        return true;
    }
    std::cout << " ROI bw check failed!" << std::endl;
    return false;
}

void GameVideoAnalyzer::assign_hero(const int& level, const cv::Point& position, std::vector<HeroStatus>* hero_status_list, const int& ts) {
    if (is_heroes_list_initialized_ == false) {
        std::cout << "Initializing heroes list..." << std::endl;
        HeroStatus hero = {hero_id_++, position, level};
        hero_status_list->push_back(hero);
        heroes_list_.push_back(hero);
        last_updated_.push_back(ts);
        appearances_.push_back(1);
    } else {
        double dist_min = 20;   // min dist threshold
        int i_min = -1;
        for (size_t i = 0; i < heroes_list_.size(); i++) {
            double dist = sqrt(pow(position.x - heroes_list_[i].position.x, 2) + 
                               pow(position.y - heroes_list_[i].position.y, 2));
            if (dist < dist_min) {
                dist_min = dist;
                i_min = i;
            }
        }
        std::cout << "Identified level " << level << " hero at " << position << ", ";
        // TODO: search all heroes whose distance is below threshold, not the minimum one.
        if ((i_min == -1) ||
            ((i_min != -1) && (ts - last_updated_[i_min] > 3000)) ||
            ((i_min != -1) && (level < heroes_list_[i_min].level))) {
            // new hero
            // don't retrieve hero after it's been missing for at least 3000ms
            HeroStatus hero = {hero_id_++, position, level};
            hero_status_list->push_back(hero);
            heroes_list_.push_back(hero);
            last_updated_.push_back(ts);
            appearances_.push_back(1);
            std::cout << "assign new hero id = " << hero.hero_id << std::endl;
        } else if (level == heroes_list_[i_min].level) {
            // existing hero
            HeroStatus hero = {heroes_list_[i_min].hero_id, position, level};
            hero_status_list->push_back(hero);
            heroes_list_[i_min].position = position;
            last_updated_[i_min] = ts;
            appearances_[i_min]++;
            std::cout << "merge old hero id = " << hero.hero_id << std::endl;
        } else if (level == heroes_list_[i_min].level + 1) {
            // existing hero, level up by 1
            HeroStatus hero = {heroes_list_[i_min].hero_id, position, level};
            hero_status_list->push_back(hero);
            heroes_list_[i_min].position = position;
            heroes_list_[i_min].level = level;
            last_updated_[i_min] = ts;
            appearances_[i_min]++;
            std::cout << "merge old hero id (levelup) = " << hero.hero_id << std::endl;
        } else {
            // Level up >1, new hero
            HeroStatus hero = {hero_id_++, position, level};
            hero_status_list->push_back(hero);
            heroes_list_.push_back(hero);
            last_updated_.push_back(ts);
            appearances_.push_back(1);
            std::cout << "assign new hero id (leveldif) = " << hero.hero_id << std::endl;
        }
    }
}

void GameVideoAnalyzer::track_hero(cv::Mat* src, std::vector<HeroStatus>* hero_status_list, const int& ts, const std::vector<cv::Mat>& number_samples, const cv::Mat& mask, const double& avg_err_thres, const size_t& bw_thres) {
    cv::Mat src_gray, src_bw, src_bw_display;
    cv::cvtColor(*src, src_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(src_gray, src_bw, bw_thres, 255, cv::THRESH_BINARY);
    cv::cvtColor(src_bw, src_bw_display, cv::COLOR_GRAY2BGR);
    // cv::imwrite("gray.bmp", src_gray);
    // cv::imwrite("bw.bmp", src_bw);

    std::vector<std::vector<cv::Point>> contours_mask;
    cv::findContours(mask, contours_mask, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    for (size_t i = 0; i < contours_mask.size(); i++) {
        cv::drawContours(src_bw_display, contours_mask, i, cv::Scalar(0, 255, 255));
    }

    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::pair<cv::Point, int>> rect_num_vec;   // [yyyxxxx] coordinate and number detected
    cv::findContours(src_bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect number_box = cv::boundingRect(contours[i]);
        // std::cout << "contour id: " << i << " bounding box: " << number_box << std::endl;
        // alternative way to capture number samples
        // if (i == 153) {
        //     cv::imwrite("../samples/l0.bmp", src_bw(number_box));
        // }

        bool masked = false;
        // filter region
        for (size_t i = 0; i < contours_mask.size() && ~masked; i++) {
            if (cv::pointPolygonTest(contours_mask[i], cv::Point(number_box.x, number_box.y), false) > 0 ||
                cv::pointPolygonTest(contours_mask[i], cv::Point(number_box.x + number_box.width, number_box.y), false) > 0 ||
                cv::pointPolygonTest(contours_mask[i], cv::Point(number_box.x, number_box.y + number_box.height), false) > 0 ||
                cv::pointPolygonTest(contours_mask[i], cv::Point(number_box.x + number_box.width, number_box.y + number_box.height), false) > 0) {
                masked = true;
            }
        }
        if (masked) {
            cv::rectangle(src_bw_display, number_box, cv::Scalar(255, 0, 255), 1);
            continue;
        }

        // size of segmented regions are restricted
        if (number_box.height > 15 || number_box.height < 12) {
            cv::rectangle(src_bw_display, number_box, cv::Scalar(255, 255, 0), 1);
            continue;
        }
        if (number_box.width > 10 || number_box.width < 4) {
            cv::rectangle(src_bw_display, number_box, cv::Scalar(255, 0, 0), 1);
            continue;
        }

        // black-and-white check
        // filter "colored" ROIs with high S and V valud in HSV space
        cv::Mat roi_colored = (*src)(number_box);
        if (is_black_white(roi_colored) == false) {
            cv::rectangle(src_bw_display, number_box, cv::Scalar(0, 100, 255), 1);
            continue;
        }

        cv::rectangle(src_bw_display, number_box, cv::Scalar(0, 255, 0), 1);
        int number_detected = GameVideoAnalyzer::detect_number_roi(&src_bw, number_box, number_samples, avg_err_thres);
        if (number_detected != -1) {
            rect_num_vec.push_back(std::pair<cv::Point, int>(cv::Point(number_box.x, number_box.y), number_detected));
            cv::rectangle(src_bw_display, number_box, cv::Scalar(0, 0, 255), 1);
            // std::cout << "detected number:" << number_detected << std::endl;
        }
    }

    // for (size_t i = 0; i < rect_num_vec.size(); i++) {
    //     std::cout << i << ": " << rect_num_vec[i].first << ", " << rect_num_vec[i].second << std::endl;
    // }

    // detect numbers in the same level icon
    std::vector<bool> is_single_digit(rect_num_vec.size(), true);
    for (size_t i = 0; i < rect_num_vec.size(); i++) {
        if (!is_single_digit[i]) {
            continue;
        }
        // for (size_t k = 0; k < is_single_digit.size(); k++) {
        //     std::cout << is_single_digit[k];
        // }
        // std::cout << std::endl;
        // std::cout << "i = " << i << std::endl;
        cv::Point p1 = rect_num_vec[i].first;
        int num1 = rect_num_vec[i].second;
        for (size_t j = i + 1; j < rect_num_vec.size(); j++) {
            // std::cout << "j = " << j << std::endl;
            cv::Point p2 = rect_num_vec[j].first;
            int num2 = rect_num_vec[j].second;
            if (abs(p1.y - p2.y) < 3 && abs(p1.x - p2.x) < 15 && abs(p1.x - p2.x) > 8) {
                int hero_level = (p1.x > p2.x) ? num1 + 10 * num2 : num2 + 10 * num1;
                // std::cout << hero_level;
                // restrict valid hero level lte 15
                if (hero_level <= 15) {
                    assign_hero(hero_level, p1.x > p2.x ? p1 : p2, hero_status_list, ts);
                    is_single_digit[i] = false;
                    is_single_digit[j] = false;
                    break;
                }
            }
        }
        if (is_single_digit[i] && num1 > 0) {
            assign_hero(num1, p1, hero_status_list, ts);
        }
    }

    if (is_heroes_list_initialized_ == false) {
        is_heroes_list_initialized_ = true;
    }
    std::cout << "Current heroes list:" << std::endl;
    std::cout << "Id\tLevel\tPosition\tLast updated\t\tAppearances" << std::endl;
    for (size_t i = 0; i < heroes_list_.size(); i++) {
        std::cout << heroes_list_[i].hero_id << '\t';
        std::cout << heroes_list_[i].level << '\t';
        std::cout << heroes_list_[i].position << '\t';
        std::cout << last_updated_[i] << '\t' << '\t';
        std::cout << appearances_[i] << std::endl;
    }

    cv::namedWindow("icon");
    cv::imshow("icon", src_bw_display);
    // cv::waitKey(0);
}

void GameVideoAnalyzer::delete_inactive_heroes(const int& ts, const int& inactive_time, const int& num_app) {
    for (size_t i = 0; i < heroes_list_.size(); i++) {
        std::vector<HeroStatus>::iterator hero_it = heroes_list_.begin() + i;
        std::vector<int>::iterator lu_it = last_updated_.begin() + i;
        std::vector<int>::iterator ap_it = appearances_.begin() + i;
        if ((ts - last_updated_[i] > inactive_time) &&
            (appearances_[i] < num_app)) {
            heroes_list_.erase(hero_it);
            last_updated_.erase(lu_it);
            appearances_.erase(ap_it);
            std::cout << "Deleted hero id " << heroes_list_[i].hero_id << " from list, last update at " << last_updated_[i] << "ms with " << appearances_[i] << " appearance(s)" << std::endl;
        }
    }
    std::cout << "Current heroes list after deleting inactive heroes:" << std::endl;
    std::cout << "Id\tLevel\tPosition\tLast updated\t\tAppearances" << std::endl;
    for (size_t i = 0; i < heroes_list_.size(); i++) {
        std::cout << heroes_list_[i].hero_id << '\t';
        std::cout << heroes_list_[i].level << '\t';
        std::cout << heroes_list_[i].position << '\t';
        std::cout << last_updated_[i] << '\t' << '\t';
        std::cout << appearances_[i] << std::endl;
    }
}

int main(int argc, char** argv) {
    std::vector<cv::String> filenames;
    cv::String folder = "/home/fyz/frames";
    cv::glob(folder, filenames);
    std::vector<FrameStatus> status_list;  // status of each frame are stored in this vector
    std::vector<double> dist;   // distances between joystick axis and center in each frame

    std::vector<cv::Mat> number_samples;
    for (size_t i = 0; i < 10; i++) {
        std::string filename = "../samples/" + std::to_string(i) + ".bmp";
        std::cout << "Loading number sample from file " << filename << std::endl;
		cv::Mat num_sample = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		if (!num_sample.data) {
			std::cerr << "Load file " << filename << "failed!\n";
			return -1;
		}
        number_samples.push_back(num_sample);
    }
    // cv::namedWindow("samples");
    // for (size_t i = 0; i < 10; i++) {
    //     cv::imshow("samples", number_samples[i]);
    //     cv::waitKey(5);
    // }

    std::vector<cv::Mat> number_samples_money;
    for (size_t i = 0; i < 10; i++) {
        std::string filename = "../samples/m" + std::to_string(i) + ".bmp";
		std::cout << "Loading number sample from file " << filename << std::endl;
		cv::Mat num_sample = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		if (!num_sample.data) {
			std::cerr << "Load file " << filename << "failed!\n";
			return -1;
		}
		number_samples_money.push_back(num_sample);
    }

    std::vector<cv::Mat> number_samples_level;
    for (size_t i = 0; i < 10; i++) {
        std::string filename = "../samples/l" + std::to_string(i) + ".bmp";
		std::cout << "Loading number sample from file " << filename << std::endl;
		cv::Mat num_sample = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		if (!num_sample.data) {
			std::cerr << "Load file " << filename << "failed!\n";
			return -1;
		}
        number_samples_level.push_back(num_sample);
    }

    // load mask.bmp used in level icon recognizion
    cv::Mat icon_mask;
    icon_mask = cv::imread("../samples/mask.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    std::cout << "Loading mask file." << std::endl;
	if (!icon_mask.data) {
		std::cerr << "Load file " << icon_mask << "failed!\n";
		return -1;
	}

    // configurations
    const double avg_err_thres_largenum = 0.3;
    const double avg_err_thres_money = 0.99;
    const size_t bw_thres_largenum = 150;
    const size_t bw_thres_smallnum = 210;
    const size_t bw_thres_level = 180;

    // icon radii
    const double radius_spell = 52.0;
    const double radius_skill = 40.0;

    GameVideoAnalyzer game_video_analyzer;

    // for (size_t i = 0; i < filenames.size(); i++) {
    for (size_t i = 141; i < 1002; i++) {
    // for (size_t i = 1740; i < 16000; i++) {
    // for (size_t i = 938; i <= 938; i++) {
        std::cout << "Reading " << filenames[i] << ".\n";
        cv::Mat src = cv::imread(filenames[i]);
        if (!src.data) {
            std::cerr << "Fail reading image!\n";
			return -1;
        }
        game_video_analyzer.adjust_size(&src);

        // static int h = src.rows;
        // static int w = src.cols;
        // // use height as the main measurement of image size
        // static double r = h / 720.0;

        FrameStatus status;
        size_t found0 = filenames[i].find_last_of("_");
        size_t found1 = filenames[i].find_last_of(".");
        int ts = static_cast<int>(std::atof(filenames[i].substr(found0 + 1, found1).c_str()) * 1000.0);
        status.ts = ts;
        std::cout << "timestamp = " << status.ts << std::endl;

        // number samples 0 - 9
        // cv::Mat number = src(cv::Rect(1161, 420 - radius_spell * 0.3, radius_spell * 0.4, radius_spell * 0.6));
        // cv::Mat number = src(cv::Rect(55, 340, 14, 20));
        // cv::Mat number_gray;
        // cv::cvtColor(number, number_gray, cv::COLOR_BGR2GRAY);
        // cv::threshold(number_gray, number_gray, 210, 255, cv::THRESH_BINARY);
        // std::vector<std::vector<cv::Point>> contours;
        // cv::findContours(number_gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        // cv::Rect number_box = cv::boundingRect(contours[0]);
        // cv::namedWindow("number");
        // cv::imshow("number", number_gray);
        // cv::imwrite("../samples/m9.bmp", number_gray(number_box));
        // cv::waitKey(0);

        // Use flexible location number detection for level icon
        std::vector<HeroStatus> hero_status_list;
        game_video_analyzer.track_hero(&src, &hero_status_list, ts, number_samples_level, icon_mask, 0.3, bw_thres_level);
        status.hero_list = hero_status_list;

        // prune heroes list
        game_video_analyzer.delete_inactive_heroes(ts, 1000, 5);

        // Use exact coordiates for spell and skill icon
        int num;
        cv::Mat src_roi;

        // money number detection
        src_roi = src(cv::Rect(18, 340, 64, 22));
        cv::rectangle(src, cv::Rect(18, 340, 64, 22), cv::Scalar(0, 0, 255), 1);
        num = game_video_analyzer.detect_number_fixed(&src_roi, number_samples_money, avg_err_thres_money, bw_thres_smallnum, cv::Vec4b(10, 16, 3, 11));
        std::cout << "Current money: " << num << std::endl;
        status.money = num;

        cv::circle(src, cv::Point(1161, 420), radius_spell, cv::Scalar(0, 0, 255), 3);
        src_roi = src(cv::Rect(1161 - radius_spell * 0.8, 420 - radius_spell * 0.4, radius_spell * 1.6, radius_spell * 0.8));
        num = game_video_analyzer.detect_number_fixed(&src_roi, number_samples, avg_err_thres_largenum, bw_thres_largenum, cv::Vec4b(0, 0, 0, 0));
        std::cout << "Spell 1 cooldown: " << num << std::endl;
        status.spell1_cd = num;

        cv::circle(src, cv::Point(1028, 497), radius_spell, cv::Scalar(0, 0, 255), 3);
        src_roi = src(cv::Rect(1028 - radius_spell * 0.8, 497 - radius_spell * 0.4, radius_spell * 1.6, radius_spell * 0.8));
        num = game_video_analyzer.detect_number_fixed(&src_roi, number_samples, avg_err_thres_largenum, bw_thres_largenum, cv::Vec4b(0, 0, 0, 0));
        std::cout << "Spell 2 cooldown: " << num << std::endl;
        status.spell2_cd = num;

        cv::circle(src, cv::Point(949, 630), radius_spell, cv::Scalar(0, 0, 255), 3);
        src_roi = src(cv::Rect(949 - radius_spell * 0.8, 630 - radius_spell * 0.4, radius_spell * 1.6, radius_spell * 0.8));
        num = game_video_analyzer.detect_number_fixed(&src_roi, number_samples, avg_err_thres_largenum, bw_thres_largenum, cv::Vec4b(0, 0, 0, 0));
        std::cout << "Spell 3 cooldown: " << num << std::endl;
        status.spell3_cd = num;

        cv::circle(src, cv::Point(643, 644), radius_skill, cv::Scalar(0, 0, 255), 3);
        src_roi = src(cv::Rect(643 - radius_skill * 0.8, 644 - radius_skill * 0.4, radius_skill * 1.6, radius_skill * 0.8));
        num = game_video_analyzer.detect_number_fixed(&src_roi, number_samples, avg_err_thres_largenum, bw_thres_largenum, cv::Vec4b(0, 0, 0, 0));
        std::cout << "Skill 1 cooldown: " << num << std::endl;
        status.skill1_cd = num;

        cv::circle(src, cv::Point(738, 644), radius_skill, cv::Scalar(0, 0, 255), 3);
        src_roi = src(cv::Rect(738 - radius_skill * 0.8, 644 - radius_skill * 0.4, radius_skill * 1.6, radius_skill * 0.8));
        num = game_video_analyzer.detect_number_fixed(&src_roi, number_samples, avg_err_thres_largenum, bw_thres_largenum, cv::Vec4b(0, 0, 0, 0));
        std::cout << "Skill 2 cooldown: " << num << std::endl;
        status.skill2_cd = num;

        cv::circle(src, cv::Point(837, 644), radius_skill, cv::Scalar(0, 0, 255), 3);
        src_roi = src(cv::Rect(837 - radius_skill * 0.8, 644 - radius_skill * 0.4, radius_skill * 1.6, radius_skill * 0.8));
        num = game_video_analyzer.detect_number_fixed(&src_roi, number_samples, avg_err_thres_largenum, bw_thres_largenum, cv::Vec4b(0, 0, 0, 0));
        std::cout << "Skill 3 cooldown: " << num << std::endl;
        status.skill3_cd = num;

        cv::circle(src, cv::Point(1155, 279), radius_skill, cv::Scalar(0, 0, 255), 3);
        src_roi = src(cv::Rect(1155 - radius_skill * 0.8, 279 - radius_skill * 0.4, radius_skill * 1.6, radius_skill * 0.8));
        num = game_video_analyzer.detect_number_fixed(&src_roi, number_samples, avg_err_thres_largenum, bw_thres_largenum, cv::Vec4b(0, 0, 0, 0));
        std::cout << "Skill 4 cooldown: " << num << std::endl;
        status.skill4_cd = num;

        // Use Hough circle detection for virtual joystick
        double joystick_angle;
        joystick_angle = game_video_analyzer.estimate_joystick_angle(&src);
        std::cout << "Joystick angle: " << joystick_angle << std::endl;
        status.joystick_angle = joystick_angle;

        // push status in this frame to status list
        game_video_analyzer.update_frame_status(status);

        // show main window
        cv::namedWindow("Video");
        cv::imshow("Video", src);
        int key = cv::waitKey(30);
        if (key == 0x1b) {
            break;
        }
        else if (key != 0xff) {
            key = cv::waitKey(0);
            if (key == 0x1b) {
                break;
            }
        }
    }
    
    double mean, stdvar;
    game_video_analyzer.estimate_js_axis_status(&mean, &stdvar);
    std::cout << "Joystick to axis length mean: " << mean << ", stdvar: " << stdvar << std::endl;
    cv::waitKey(0);

    return 0;
}