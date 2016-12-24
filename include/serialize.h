#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__


#include "json.hpp"

namespace neat
{

class ISerialize
{
public:
    virtual nlohmann::json serialize() const = 0;
};


void serialize_to_file(std::string path, const ISerialize& object, bool pretty=true);

};
#endif
