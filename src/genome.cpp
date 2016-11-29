#include <cassert>
#include <iostream>

#include "cpplinq.hpp"

#include "genome.h"

namespace neat
{

//=========================== Constructors ===================================
Genome::Genome(GenomeID id, std::size_t inputs, std::size_t outputs):m_genome_id(id),
                                               m_fitness(0),
                                               m_adjusted_fitness(0),
                                               m_num_inputs(inputs),
                                               m_num_outputs(outputs),
                                               m_amount_to_spawn(0),
                                               m_species_id(0),
                                               m_random()
{
    double input_row_slice = 1 / (double)(inputs+2);

    m_neuron_genes = cpplinq::range(0, inputs)
        >> cpplinq::select([&](int i){ return NeuronGene(NeuronType::INPUT,
                                                        i,
                                                        0,
                                                        (i+2)*input_row_slice);
                                    })
        >> cpplinq::to_vector();

    m_neuron_genes.push_back(NeuronGene(NeuronType::BIAS, inputs, 0, input_row_slice));

    double output_row_slice = 1 / (double)(outputs+1);

    auto output_neurons = cpplinq::range(1, outputs)
        >> cpplinq::select([&](int i) { return NeuronGene(NeuronType::OUTPUT,
                                                          i + inputs + 1,
                                                          1,
                                                          (i+1)*output_row_slice);
                                      })
        >> cpplinq::to_vector();
    m_neuron_genes.insert(m_neuron_genes.end(), output_neurons.begin(), output_neurons.end());

    for(int i = 0; i < inputs + 1; ++i)
    {
        for(int j = inputs + 1; j < inputs + outputs + 1; ++j)
        {
            int innovation_id = inputs + outputs + NumGenes();
            m_link_genes.push_back(LinkGene(m_neuron_genes[i].ID,
                                            m_neuron_genes[j].ID,
                                            m_random.RandomClamped(-1.0, 1.0),
                                            true,
                                            innovation_id));
        }
    }
}


Genome::Genome(GenomeID id,
                std::vector<NeuronGene> neuron_genes,
                std::vector<LinkGene> link_genes,
                std::size_t num_inputs,
                std::size_t num_outputs):m_genome_id(id),
                                         m_neuron_genes(neuron_genes),
                                         m_link_genes(link_genes),
                                         m_fitness(0),
                                         m_adjusted_fitness(0),
                                         m_amount_to_spawn(0),
                                         m_num_inputs(num_inputs),
                                         m_num_outputs(num_outputs),
                                         m_random()
{}

Genome::~Genome()
{
}


//================================ PUBLIC METHODS =================================
void Genome::AddNeuron(double mutation_prob,
                       InnovationDB& inno_db,
                       int num_trys_to_find_old_link)
{
    if(m_random.RandomDouble() > mutation_prob)
    {
        return;
    }

    auto find_link_idx = [&](int upper_bound)
    {
        int link_idx = m_random.RandomClamped(0, upper_bound);
        const LinkGene& link = m_link_genes[link_idx];
        const NeuronGene& neuron = m_neuron_genes[GetNeuronIndex(link.FromNeuronID)];
        if(link.IsEnabled &&
            !link.IsRecurrent &&
            neuron.Type != NeuronType::BIAS)
        {
            return link_idx;
        }
        else
        {
            return -1;
        }
    };

    int chosen_link = -1;
    // if the genome is small then prefer older links to avoid a chaining
    // effect.
    const int SizeThreshold = m_num_inputs + m_num_outputs + 5;
    if(m_neuron_genes.size() < SizeThreshold)
    {
        int upper_bound = m_neuron_genes.size() - 1 - (int)std::sqrt(m_neuron_genes.size());
        while(num_trys_to_find_old_link-- && chosen_link < 0)
        {
            chosen_link = find_link_idx(upper_bound);
        }
        // failed to find a good old link
        if(chosen_link == -1) return;
    }
    else
    {
        while(chosen_link == -1)
        {
            chosen_link = find_link_idx(m_neuron_genes.size());
        }
    }

    auto& link = m_link_genes[chosen_link];
    // disable the link
    link.IsEnabled = false;
    double original_weight = link.Weight;

    // identify connected neurons
    NeuronID from = link.FromNeuronID;
    NeuronID to = link.ToNeuronID;

    // calculate the depth and width of the new neuron. We can use the depth to
    // see if the link feeds backwards or forwards
    auto& from_neuron = m_neuron_genes[GetNeuronIndex(from)];
    auto& to_neuron =  m_neuron_genes[GetNeuronIndex(to)];
    double new_depth = (from_neuron.SplitY + to_neuron.SplitY) / 2;
    double new_width = (from_neuron.SplitX + to_neuron.SplitX) / 2;

    auto inno_id = inno_db.GetInnovationId(from, to, InnovationType::NEW_NEURON);

  /**
   * It is possible for NEAT to repeatedly do the following:
   * 1. Find a link. Lets say we choose link 1 to 5
   * 2. Disable the link,
   * 3. Add a new neuron and two new links
   * 4. The link disabled in Step 2 maybe re-enabled when this genome
   * is recombined with a genome that has that link enabled.
   * 5  etc etc
   *
   * Therefore, this function must check to see if a neuron ID is already
   * being used. If it is then the function creates a new innovation
   * for the neuron.
   */
    if(inno_id >= 0)
    {
        auto neuron_id = inno_db.GetNeuronID(inno_id);
        auto neuron_id_match = [neuron_id](const NeuronGene& neuron)
        {
            return neuron.ID == neuron_id;
        };

        if(cpplinq::from(m_neuron_genes) >> cpplinq::any(neuron_id_match))
        {
            inno_id = -1;
        }
    }

    if(inno_id < 0)
    {
        NeuronID neuron_id = inno_db.AddNeuronInnovation(from_neuron.ID,
                                                         to_neuron.ID,
                                                         NeuronType::HIDDEN,
                                                         new_width,
                                                         new_depth);
        NeuronGene ng(NeuronType::HIDDEN,
                      neuron_id,
                      new_depth,
                      new_width);
        m_neuron_genes.push_back(ng);

        InnovationID link_id = inno_db.AddLinkInnovation(from, neuron_id);
        LinkGene bottom_link(from,
                             neuron_id,
                             m_random.RandomDouble(),
                             true,
                             link_id);
        m_link_genes.push_back(bottom_link);

        link_id = inno_db.AddLinkInnovation(neuron_id, to);
        LinkGene top_link(neuron_id,
                          to,
                          m_random.RandomDouble(),
                          true,
                          link_id);
        m_link_genes.push_back(top_link);

    }
    else
    {
        NeuronID neuron_id = inno_db.GetNeuronID(inno_id);
        InnovationID link_bottom_id = inno_db.GetInnovationId(from, neuron_id, InnovationType::NEW_LINK);
        InnovationID link_top_id = inno_db.GetInnovationId(neuron_id, to, InnovationType::NEW_LINK);

        assert(link_bottom_id >= 0 && link_top_id >= 0);

        LinkGene link_bottom(from, neuron_id, 1.0, true, link_bottom_id);
        LinkGene link_top(neuron_id, to, original_weight, true, link_top_id);
        m_link_genes.push_back(link_bottom);
        m_link_genes.push_back(link_top);

        NeuronGene new_neuron(NeuronType::HIDDEN, neuron_id, new_depth, new_width);
        m_neuron_genes.push_back(new_neuron);
    }
}

bool Genome::AddLink(double mutation_prob,
                     double recurrent_prob,
                     InnovationDB& innovationDB,
                     int num_trys_recurrent,
                     int num_trys_add_link)
{
    if(m_random.RandomDouble() > mutation_prob)
    {
        return false;
    }

    NeuronID neuron_id_from = -1;
    NeuronID neuron_id_to = -1;

    // if unable to find any suitable neurons then return
    if(!FindNonRecurrentNeuron(neuron_id_from, neuron_id_to, recurrent_prob, num_trys_recurrent) &&
       !FindUnlinkedNeurons(neuron_id_from, neuron_id_to, num_trys_recurrent))
    {
        std::cout << "Could not find any suitable neurons to link" << std::endl;
        return false;
    }
    else
    {
        assert(neuron_id_from >=0 && neuron_id_to >= 0);

        auto is_recurrent_link = [&](NeuronID neuron_id1, NeuronID neuron_id2) {
            auto& neuron1 = m_neuron_genes[GetNeuronIndex(neuron_id1)];
            auto& neuron2 = m_neuron_genes[GetNeuronIndex(neuron_id2)];
            return neuron_id1 == neuron_id2 || neuron1.SplitY > neuron2.SplitY;
        };
        InnovationID innovation_id = innovationDB.GetInnovationId(neuron_id_from, neuron_id_to, InnovationType::NEW_LINK);
        bool is_recurrent = is_recurrent_link(neuron_id_from, neuron_id_to);

        innovation_id = innovation_id < 0
            ? innovationDB.AddLinkInnovation(neuron_id_from, neuron_id_to)
            : innovation_id;

        LinkGene new_link(neuron_id_from,
                         neuron_id_to,
                         m_random.RandomClamped<double>(),
                         true,
                         innovation_id,
                         is_recurrent);
        m_link_genes.push_back(new_link);
        return true;
    }
}


void Genome::MutateWeights(double mutation_prob, double prob_new_weight, double max_perturbation)
{
    for(auto& link_gene : m_link_genes)
    {
        if(m_random.RandomDouble() < mutation_prob)
        {
            if(m_random.RandomDouble() < prob_new_weight)
            {
                link_gene.Weight = m_random.RandomClamped(-1.0, 1.0);
            }
            else
            {
                link_gene.Weight += m_random.RandomDouble() * max_perturbation;
            }
        }
    }
}


void Genome::MutateActivationResponse(double mutation_prob, double max_perturbation)
{
    for(auto& neuron_gene : m_neuron_genes)
    {
        if(m_random.RandomDouble() < mutation_prob)
        {
            neuron_gene.ActivationResponse += m_random.RandomClamped(-1.0, 1.0) * max_perturbation;
        }
    }
}


double Genome::CalculateCompatabilityScore(const Genome& other) const
{
    //travel down the length of each genome counting the number of
    //disjoint genes, the number of excess genes and the number of
    //matched genes
    double num_disjoint = 0;
    double num_matched  = 0;
    double num_excess   = std::fabs(NumLinks() - other.NumLinks());

    //this records the summed difference of weights in matched genes
    double weight_difference = 0;

    //position holders for each genome. They are incremented as we
    //step down each genomes length.
    int g1 = 0;
    int g2 = 0;

    // REMARK: This is different from Matt's code
    while ( (g1 < NumLinks()) && (g2 < other.NumLinks()) )
    {
        //get innovation numbers for each gene at this point
        InnovationID id1 = m_link_genes[g1].InnovID;
        InnovationID id2 = other.m_link_genes[g2].InnovID;

        //innovation numbers are identical so increase the matched score
        if (id1 == id2)
        {
            ++g1;
            ++g2;
            ++num_matched;

            //get the weight difference between these two genes
            weight_difference += std::fabs(m_link_genes[g1].Weight - other.m_link_genes[g2].Weight);
        }

        //innovation numbers are different so increment the disjoint score
        if (id1 < id2)
        {
            ++num_disjoint;
            ++g1;
        }

        if (id1 > id2)
        {
            ++num_disjoint;
            ++g2;
        }

    }//end while

    //get the length of the longest genome
    int longest = std::max(other.NumGenes(), NumGenes());

    //these are multipliers used to tweak the final score.
    const double mDisjoint = 1;
    const double mExcess   = 1;
    const double mMatched  = 0.4;

    //finally calculate the scores
    double score = (mExcess * num_excess/(double)longest) +
                   (mDisjoint * num_disjoint/(double)longest) +
                   (mMatched * weight_difference / num_matched);

    return score;
}


Genome Genome::Crossover(const Genome& mom, const InnovationDB& inno_db, GenomeID genome_id)
{
    enum PARENT_TYPE{MOM, DAD};
    Genome& dad = *this;
    PARENT_TYPE best;

    if(mom.Fitness() == dad.Fitness())
    {
        if(mom.NumLinks() == dad.NumLinks())
        {
            best = (PARENT_TYPE)m_random.RandomClamped(0, 1);
        }
        else
        {
            best = mom.NumLinks() < dad.NumLinks() ? MOM : DAD;
        }
    }
    else
    {
        best = mom.Fitness() > dad.Fitness() ? MOM : DAD;
    }

    std::vector<NeuronGene> baby_neurons;
    std::vector<LinkGene> baby_links;

    std::vector<NeuronID> neuron_ids;

    auto current_mom = mom.m_link_genes.begin();
    auto current_dad = dad.m_link_genes.begin();

    auto mom_end = mom.m_link_genes.end();
    auto dad_end = dad.m_link_genes.end();
    LinkGene selected_gene;

    while(!((current_mom == mom_end) && current_dad == dad_end))
    {
        if(current_mom == mom_end && current_dad != dad_end)
        {
            if(best == DAD)
            {
                selected_gene = *current_dad;
            }
            ++current_dad;
        }
        else if(current_mom != mom_end && current_dad == dad_end)
        {
            if(best == MOM)
            {
                selected_gene = *current_mom;
            }
            ++current_mom;
        }
        else if(current_mom->InnovID < current_dad->InnovID)
        {
            if(best == MOM)
            {
                selected_gene = *current_mom;
            }
            ++current_mom;
        }
        else if(current_dad->InnovID < current_mom->InnovID)
        {
            if(best == DAD)
            {
                selected_gene = *current_dad;
            }
            ++current_dad;
        }
        else if(current_dad->InnovID == current_mom->InnovID)
        {
            if(m_random.RandomClamped(0.0, 1.0) < 0.5)
            {
                selected_gene = *current_mom;
            }
            else
            {
                selected_gene = *current_dad;
            }
            ++current_dad;
            ++current_mom;
        }

        if(!baby_links.size())
        {
            baby_links.push_back(selected_gene);
        }
        else
        {
            std::size_t size = baby_links.size();
            if(baby_links[size-1].InnovID != selected_gene.InnovID)
            {
                baby_links.push_back(selected_gene);
            }
        }

        if(!(cpplinq::from(neuron_ids) >> cpplinq::contains(selected_gene.FromNeuronID)))
        {
            neuron_ids.push_back(selected_gene.FromNeuronID);
        }

        if(!(cpplinq::from(neuron_ids) >> cpplinq::contains(selected_gene.ToNeuronID)))
        {
            neuron_ids.push_back(selected_gene.ToNeuronID);
        }

        std::sort(neuron_ids.begin(), neuron_ids.end());
    }

    for(NeuronID nid : neuron_ids)
    {
        baby_neurons.push_back(inno_db.CloneNeuronFromID(nid));
    }

    Genome baby(genome_id,
                baby_neurons,
                baby_links,
                m_num_inputs,
                m_num_outputs);
    return baby;
}

void Genome::SortLinks()
{
    std::sort(m_link_genes.begin(), m_link_genes.end());
}


//========================================== PRIVATE METHODS ===============================
bool Genome::FindNonRecurrentNeuron(NeuronID& neuron_id_from, NeuronID& neuron_id_to, double prob, int num_trys)
{
    bool result = false;
    auto is_acceptable = [&](NeuronGene& neuron)
    {
        return !neuron.IsRecurrent
               && neuron.Type != NeuronType::BIAS
               && neuron.Type != NeuronType::INPUT;
    };

    if(m_random.RandomDouble() < prob)
    {
        while(num_trys--)
        {
            // get random neuron
            int neuron_idx = m_random.RandomClamped(m_num_inputs+1, m_neuron_genes.size()-1);
            auto& neuron = m_neuron_genes[neuron_idx];

            if(is_acceptable(neuron))
            {
                neuron_id_from = neuron_id_to = neuron.ID;
                neuron.IsRecurrent = true;
                result = true;
                break;
            }
        }
    }
    return result;
}

bool Genome::FindUnlinkedNeurons(NeuronID& neuron_id_from, NeuronID& neuron_id_to, int num_trys)
{
    bool result = false;
    auto is_acceptable = [&](NeuronGene& neuron_from, NeuronGene& neuron_to)
    {
        return !IsExistingLink(neuron_from.ID, neuron_to.ID)
               && neuron_from.ID != neuron_to.ID
               && neuron_to.Type != NeuronType::BIAS
               && neuron_to.Type != NeuronType::INPUT;
    };

    while(num_trys--)
    {
        int from_idx = m_random.RandomClamped((std::size_t)0, m_neuron_genes.size()-1);
        // can't link back to input neurons, so skipping them in idx generation
        int to_idx = m_random.RandomClamped(m_num_inputs+1, m_neuron_genes.size()-1);

        auto& neuron_from = m_neuron_genes[from_idx];
        auto& neuron_to = m_neuron_genes[to_idx];

        if(is_acceptable(neuron_from, neuron_to))
        {
            neuron_id_from = neuron_from.ID;
            neuron_id_to = neuron_to.ID;
            result = true;
            break;
        }
    }

    return result;
}

bool Genome::IsExistingLink(NeuronID neuron_from_id, NeuronID neuron_to_id)
{
    return cpplinq::from(m_link_genes)
        >> cpplinq::any([&](const LinkGene& link)
        {
            return link.FromNeuronID == neuron_from_id
                && link.ToNeuronID == neuron_to_id;
        });
}


int Genome::GetNeuronIndex(NeuronID neuron_id)
{
    for(int i = 0; i < m_neuron_genes.size(); ++i)
    {
        if(m_neuron_genes[i].ID == neuron_id)
        {
            return i;
        }
    }
    return -1;
}

}
