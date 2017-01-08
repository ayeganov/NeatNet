#include <map>
#include <set>

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
const int IMG_HEIGHT = 500;
const int IMG_WIDTH = 1200;
const double RADIUS = 10;


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


std::map<NeuronID, cv::Point> draw_neurons(cv::Mat& image, std::map<Level, Layer>& layers, bool draw_input)
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

        // input drawing check
        if(!draw_input && depth == 0) continue;

        double layer_num = depth_size.second;
        auto& layer = layers[depth];
        double x_increment = static_cast<double>(IMG_WIDTH) / (layer.size() + 1);
        double y_increment = static_cast<double>(IMG_HEIGHT) / (layers.size() + 1);

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

            std::string neuron_id_txt = std::to_string(neuron.ID);
            putText(image, neuron_id_txt, position, cv::FONT_HERSHEY_SIMPLEX, 0.3, TEXT_CLR, 1, CV_AA);

            circle(image, position, RADIUS, NEURON_CLR, 1, CV_AA);
        }
    }

    return std::move(neuron_positions);
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
                    from_position.x += RADIUS;
                    to_position.x += RADIUS;
                }
            }
            else
            {
                // draw regular links on the left
                from_position.x -= RADIUS;
                to_position.x -= RADIUS;
            }

            cv::line(image, from_position, to_position, link_color, 1, CV_AA);
        }
    }
}

cv::Mat visualize_net(const NeuralNet& nn, bool draw_input)
{
    using namespace cpplinq;
    cv::Mat image = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);
    image.setTo(BACKGROUND_CLR);

    auto layers = get_layers(nn.GetNeurons());

    std::map<NeuronID, cv::Point> neuron_positions = draw_neurons(image, layers, draw_input);

    draw_links(image, nn.GetNeurons(), neuron_positions, draw_input);

    return image;
}

cv::Mat visualize_net(SNeuralNetPtr pnn, bool draw_input)
{
    return visualize_net(*pnn.get(), draw_input);
}


};
