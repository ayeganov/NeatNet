#include <fstream>

#include "serialize.h"
#include "utils.h"

namespace neat
{


void serialize_to_file(std::string path, const ISerialize& object, bool pretty)
{
    std::ofstream out(path);
    if(pretty)
    {
        out << object.serialize().dump(2);
    }
    else
    {
        out << object.serialize().dump();
    }
    out.close();
}


nlohmann::json deserialize_from_file(std::string path)
{
    if(!Utils::is_file_exist(path))
    {
        throw new std::ios_base::failure("Can't deserialize - file '" + path + "' doesn't exist");
    }

    std::ifstream ifs(path);
    nlohmann::json object;
    ifs >> object;
    return std::move(object);
}


};
