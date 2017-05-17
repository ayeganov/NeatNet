#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "genome.h"
#include "phenotype.h"
#include "netvisualize.h"


using namespace cv;


SCENARIO("Create an image of a neural network")
{
    GIVEN("A neural network")
    {
        neat::Params p;
        neat::Genome g(1, 3, 2, &p);
        neat::InnovationDB db(g.NeuronGenes(), g.NeuronLinks());
        g.ConnectNeurons(5, 5, db);
        g.ConnectNeurons(6, 3, db);
        neat::NeuralNet nn(g);

        WHEN("Neural network is visualized and test image is read")
        {
            bool draw_input = true;
            auto image = visualize_net(nn, 300, 200, draw_input);
            cv::imwrite("new_image.png", image);
            auto test_image = cv::imread("test_image.png");

            THEN("The generated image is identical to the reference test image")
            {
                cv::Mat image_diff1 = image - test_image;
                cv::Mat image_diff2 = test_image - image;
                double total_sum = sum(image_diff1)[0] + sum(image_diff2)[0];
                REQUIRE(total_sum == 0.0);
            }
        }
    }
}
