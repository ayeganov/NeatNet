#ifndef __PARAMS_H__
#define __PARAMS_H__

#include <cstddef>

#include "json.hpp"

namespace neat
{

class Params
{
private:
    std::size_t m_num_gens_allowed_no_improv;
    std::size_t m_num_best_genomes;
    std::size_t m_num_add_link_attempts;
    std::size_t m_num_find_old_link_attempts;
    std::size_t m_num_add_recur_link_attempts;
    std::size_t m_population_size;
    std::size_t m_max_neurons;
    std::size_t m_young_bonus_threshold;
    std::size_t m_old_penalty_threshold;

    double m_survival_rate;
    double m_compatibility_threshold;
    double m_add_neuron_chance;
    double m_add_link_chance;
    double m_add_recur_link_chance;
    double m_mutation_chance;
    double m_crossover_chance;
    double m_max_perturbation;
    double m_new_weight_chance;
    double m_activation_mutation_chance;
    double m_max_activation_perturbation;
    double m_old_penalty_scaler;
    double m_young_bonus_scaler;

    // scoring used for speciation
    double m_disjoint_scaler;
    double m_excess_scaler;
    double m_match_scaler;

public:
    Params();
    Params(std::string config_path);
    Params(const Params& params);


    // Getters
    std::size_t NumGensAllowedNoImprov() const { return m_num_gens_allowed_no_improv; }
    std::size_t NumBestGenomes() const { return m_num_best_genomes; }
    std::size_t NumAddLinkAttempts() const { return m_num_add_link_attempts; }
    std::size_t NumFindOldLinkAttempts() const { return m_num_find_old_link_attempts; }
    std::size_t NumAddRecurLinkAttempts() const { return m_num_add_recur_link_attempts; }
    std::size_t PopulationSize() const { return m_population_size; }
    std::size_t MaxNeurons() const { return m_max_neurons; }
    std::size_t YoungBonusThreshold() const { return m_young_bonus_threshold; }
    std::size_t OldPenaltyThreshold() const { return m_old_penalty_threshold; }

    double SurvivalRate() const { return m_survival_rate; }
    double CompatibilityThreshold() const { return m_compatibility_threshold; }
    double AddNeuronChance() const { return m_add_neuron_chance; }
    double AddLinkChance() const { return m_add_link_chance; }
    double AddRecurLinkChance() const { return m_add_recur_link_chance; }
    double MutationChance() const { return m_mutation_chance; }
    double CrossoverChance() const { return m_crossover_chance; }
    double MaxPerturbation() const { return m_max_perturbation; }
    double NewWeightChance() const { return m_new_weight_chance; }
    double ActivationMutationChance() const { return m_activation_mutation_chance; }
    double MaxActivationPerturbation() const { return m_max_activation_perturbation; }
    double OldPenaltyScaler() const { return m_old_penalty_scaler; }
    double YoungBonusScaler() const { return m_young_bonus_scaler; }
    double DisjointScaler() const { return m_disjoint_scaler; }
    double ExcessScaler() const { return m_excess_scaler; }
    double MatchScaler() const { return m_match_scaler; }
};

};

#endif
