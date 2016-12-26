#include <string>
#include <fstream>

#include "params.h"
#include "utils.h"


namespace neat
{

/**
 * Default constructor sets values that will work well for XOR-problem.
 */
Params::Params()
{
    m_activation_mutation_chance = 0.1;
    m_add_link_chance = 0.3;
    m_add_neuron_chance = 0.05;
    m_add_recur_link_chance = 0.05;
    m_compatibility_threshold = 0.25;
    m_crossover_chance = 0.75;
    m_disjoint_scaler = 1.0;
    m_excess_scaler = 1.0;
    m_match_scaler = 0.4;
    m_max_activation_perturbation = 0.1;
    m_max_neurons = 100;
    m_max_perturbation = 0.5;
    m_mutation_chance = 0.8;
    m_new_weight_chance = 0.1;
    m_num_add_link_attempts = 5;
    m_num_add_recur_link_attempts = 5;
    m_num_best_genomes = 5;
    m_num_find_old_link_attempts = 5;
    m_num_gens_allowed_no_improv = 15;
    m_old_penalty_scaler = 0.7;
    m_old_penalty_threshold = 10;
    m_population_size = 150;
    m_survival_rate = 0.5;
    m_young_bonus_scaler = 1.3;
    m_young_bonus_threshold = 5;
}

Params::Params(const Params& params)
{
    m_activation_mutation_chance = params.m_activation_mutation_chance;
    m_add_link_chance = params.m_add_link_chance;
    m_add_neuron_chance = params.m_add_neuron_chance;
    m_add_recur_link_chance = params.m_add_recur_link_chance;
    m_compatibility_threshold = params.m_compatibility_threshold;
    m_crossover_chance = params.m_crossover_chance;
    m_disjoint_scaler = params.m_disjoint_scaler;
    m_excess_scaler = params.m_excess_scaler;
    m_match_scaler = params.m_match_scaler;
    m_max_activation_perturbation = params.m_activation_mutation_chance;
    m_max_neurons = params.m_max_neurons;
    m_max_perturbation = params.m_max_perturbation;
    m_mutation_chance = params.m_mutation_chance;
    m_new_weight_chance = params.m_new_weight_chance;
    m_num_add_link_attempts = params.m_num_add_link_attempts;
    m_num_add_recur_link_attempts = params.m_num_add_recur_link_attempts;
    m_num_best_genomes = params.m_num_best_genomes;
    m_num_find_old_link_attempts = params.m_num_find_old_link_attempts;
    m_num_gens_allowed_no_improv = params.m_num_gens_allowed_no_improv;
    m_old_penalty_scaler = params.m_old_penalty_scaler;
    m_old_penalty_threshold = params.m_old_penalty_threshold;
    m_population_size = params.m_population_size;
    m_survival_rate = params.m_survival_rate;
    m_young_bonus_scaler = params.m_young_bonus_scaler;
    m_young_bonus_threshold = params.m_young_bonus_threshold;
}


template <typename T>
void set_value(const nlohmann::json& config, std::string name, T& value)
{
    try
    {
        auto param_value = config.value(name, value);
        if(param_value < 0)
        {
            throw new std::invalid_argument("Parameters must contain only positive values.");
        }
        value = param_value;
    }
    catch(std::out_of_range)
    {
        std::cout << "Config doesn't have param " + name << std::endl;
    }
}

Params::Params(const std::string& config_path): Params()
{
    using json = nlohmann::json;

    if(!Utils::is_file_exist(config_path))
    {
        throw new std::invalid_argument("Path " + config_path + " doesn't exist");
    }
    std::fstream in(config_path);
    json config = json::parse(in);
    InitValues(config);
}

Params::Params(nlohmann::json& json_obj): Params()
{
    InitValues(json_obj);
}

Params Params::FromString(std::string params)
{
    using json = nlohmann::json;
    json json_obj = json::parse(params);
    Params p(json_obj);
    return p;
}


// ============================ PRIVATE METHODS =====================================
void Params::InitValues(nlohmann::json& config)
{
    set_value(config, "ActivationMutationRate", m_activation_mutation_chance);
    set_value(config, "ChanceAddLink", m_add_link_chance);
    set_value(config, "ChanceAddNeuron", m_add_neuron_chance);
    set_value(config, "ChanceAddRecurrentLink", m_add_recur_link_chance);
    set_value(config, "CompatibilityThreshold", m_compatibility_threshold);
    set_value(config, "CrossoverRate", m_crossover_chance);
    set_value(config, "DisjointScaler", m_disjoint_scaler);
    set_value(config, "ExcessScaler", m_excess_scaler);
    set_value(config, "MatchScaler", m_match_scaler);
    set_value(config, "MaxActivationPerturbation", m_max_activation_perturbation);
    set_value(config, "MaxPermittedNeurons", m_max_neurons);
    set_value(config, "MaxWeightPerturbation", m_max_perturbation);
    set_value(config, "MutationProbability", m_mutation_chance);
    set_value(config, "NumAddLinkAttempts", m_num_add_link_attempts);
    set_value(config, "NumAddRecurLinkAttempts", m_num_add_recur_link_attempts);
    set_value(config, "NumBestGenomes", m_num_best_genomes);
    set_value(config, "NumFindOldLinkAttempts", m_num_find_old_link_attempts);
    set_value(config, "NumGensAllowedNoImprovement", m_num_gens_allowed_no_improv);
    set_value(config, "OldAgePenalty", m_old_penalty_scaler);
    set_value(config, "OldAgeThreshold", m_old_penalty_threshold);
    set_value(config, "PopulationSize", m_population_size);
    set_value(config, "SurvivalRate", m_survival_rate);
    set_value(config, "WeightReplacedProbability", m_new_weight_chance);
    set_value(config, "YoungBonusAgeThreshhold", m_young_bonus_threshold);
    set_value(config, "YoungFitnessBonus", m_young_bonus_scaler);
}


};
