#ifndef __NETVISUALIZE_H__
#define __NETVISUALIZE_H__

#include <opencv2/core/core.hpp>

#include "phenotype.h"

namespace neat
{


using Level = double;
using Layer = std::vector<Neuron>;


cv::Mat visualize_net(SNeuralNetPtr nn, int width, int height, bool draw_input=true);
cv::Mat visualize_net(const NeuralNet& nn, int width, int height, bool draw_input=true);


};


#endif
