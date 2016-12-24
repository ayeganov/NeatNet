#include <fstream>

#include "serialize.h"

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


};
