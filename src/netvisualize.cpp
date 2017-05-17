#include <map>
#include <set>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "netvisualize.h"

#include "cpplinq.hpp"


namespace neat
{

const cv::Scalar BACKGROUND_CLR(223, 240, 162);
const cv::Scalar NEURON_CLR(0, 0, 255);
const cv::Scalar TEXT_CLR(255, 0, 0);
const cv::Scalar FORWARD_LINK_CLR(110, 220, 0);
const cv::Scalar RECUR_LINK_CLR(0, 0, 200);
const int IMG_HEIGHT_MIN = 100;
const int IMG_WIDTH_MIN = 100;
const double RADIUS = 3;


void traverse_neural_chain(std::map<Level, Layer>& layers,
                           std::set<NeuronID>& visited,
                           const Neuron& n)
{
    // if this neuron has not been seen - traverse it
    if(!visited.count(n.ID))
    {
        visited.insert(n.ID);
        auto found = layers.find(n.SplitY);

        if(found == layers.end())
        {
            layers[n.SplitY] = std::vector<Neuron>{n};
        }
        else
        {
            found->second.push_back(n);
        }

        for(auto& link : n.OutLinks)
        {
            traverse_neural_chain(layers, visited, *link.Out);
        }
    }
}

std::map<Level, Layer> get_layers(const std::vector<Neuron>& neurons)
{
    using namespace cpplinq;
    std::map<Level, Layer> layers;
    std::set<NeuronID> visited;

    auto start_neurons = from(neurons) >> where([](const Neuron& n)
        {
            return n.Type == NeuronType::BIAS || n.Type == NeuronType::INPUT;
        })
    >> to_vector();

    for(auto& n : start_neurons)
    {
        traverse_neural_chain(layers, visited, n);
    }

    return std::move(layers);
}


std::map<NeuronID, cv::Point> calc_neuron_positions(std::map<Level, Layer>& layers, int width, int height, bool include_input)
{
    using namespace cpplinq;

    std::map<NeuronID, cv::Point> neuron_positions;

    auto depths = from(layers)
        >> select([](const std::pair<Level, Layer>& p)
        {
            return p.first;
        })
        >> to_vector();
    std::sort(depths.begin(), depths.end());

    auto depth_sizes = from(depths) >> zip_with(range(0, layers.size())) >> to_vector();

    for(auto depth_size : depth_sizes)
    {
        double depth = depth_size.first;

        // Depth 0 contains all input neurons - ignore them if input is set to
        // be ignored
        if(depth == 0 && !include_input) continue;

        double layer_num = depth_size.second;
        auto& layer = layers[depth];
        double x_increment = static_cast<double>(width) / (layer.size() + 1);
        double y_increment = static_cast<double>(height) / (layers.size() + 1);

        auto x_positions = range(1, layer.size()) >> select([&x_increment](int neuron_num)
            {
                return neuron_num * x_increment;
            })
        >> to_vector();

        auto y_position = y_increment * layer_num + RADIUS;

        auto pos_to_neuron = from(x_positions) >> zip_with(from(layer)) >> to_vector();
        for(auto x_pos_neuron : pos_to_neuron)
        {
            auto x_pos = x_pos_neuron.first;
            auto& neuron = x_pos_neuron.second;

            cv::Point position(x_pos, y_position);
            neuron_positions[neuron.ID] = position;
        }
    }

    return neuron_positions;
}


void draw_links(cv::Mat& image, const std::vector<Neuron>& neurons, std::map<NeuronID, cv::Point>& neuron_positions, bool draw_input)
{
    for(const auto& neuron : neurons)
    {
        if(!draw_input && (neuron.Type == NeuronType::INPUT || neuron.Type == NeuronType::BIAS))
        {
            continue;
        }

        for(auto& link : neuron.OutLinks)
        {
            cv::Scalar link_color = FORWARD_LINK_CLR;
            auto from_position = neuron_positions[neuron.ID];
            auto to_position = neuron_positions[link.Out->ID];

            if(link.IsRecurrent)
            {
                // is this self recurrent?
                if(link.In->ID == link.Out->ID)
                {
                    cv::Point r_pos(from_position.x - (int)RADIUS*2, from_position.y);
                    putText(image, "R", r_pos, cv::FONT_HERSHEY_SIMPLEX, 0.3, TEXT_CLR, 1, CV_AA);
                }
                // recurrent by sending output to neuron in previous layer
                else
                {
                    // draw recurrent links on the right
                    link_color = RECUR_LINK_CLR;
//                    from_position.x += RADIUS;
//                    to_position.x += RADIUS;
                }
            }
            else
            {
                // draw regular links on the left
//                from_position.x -= RADIUS;
//                to_position.x -= RADIUS;
            }

            cv::line(image, from_position, to_position, link_color, 1, CV_AA);
        }
    }
}


void draw_neurons(cv::Mat& image, std::map<NeuronID, cv::Point> neuron_positions)
{
    for(auto it = neuron_positions.begin(); it != neuron_positions.end(); ++it)
    {
//        std::string neuron_id_txt = std::to_string(it->first);
//        putText(image, neuron_id_txt, it->second, cv::FONT_HERSHEY_SIMPLEX, 0.3, TEXT_CLR, 1, CV_AA);

        circle(image, it->second, RADIUS, NEURON_CLR, CV_FILLED, CV_AA);
    }
}


cv::Mat visualize_net(const NeuralNet& nn, int width, int height, bool draw_input)
{
    using namespace cpplinq;
    if(width < IMG_WIDTH_MIN)
    {
        throw new std::invalid_argument("Image width is too small: " + std::to_string(width) + ", minimum: " + std::to_string(IMG_WIDTH_MIN));
    }

    if(height < IMG_HEIGHT_MIN)
    {
        throw new std::invalid_argument("Image height is too small: " + std::to_string(width) + ", minimum: " + std::to_string(IMG_HEIGHT_MIN));
    }

    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    image.setTo(BACKGROUND_CLR);

    auto layers = get_layers(nn.GetNeurons());

    std::map<NeuronID, cv::Point> neuron_positions = calc_neuron_positions(layers, width, height, draw_input);

    draw_links(image, nn.GetNeurons(), neuron_positions, draw_input);
    draw_neurons(image, neuron_positions);

    return image;
}

cv::Mat visualize_net(SNeuralNetPtr pnn, int width, int height, bool draw_input)
{
    return visualize_net(*pnn.get(), width, height, draw_input);
}


};
