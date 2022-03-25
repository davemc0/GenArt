#pragma once

#include <string>

inline void CopyFile(const std::string& inFName, const std::string& outFName)
{
    std::cerr << "Copying from " << inFName << " to " << outFName << '\n';
    std::string cmd = std::string("COPY /Y /V \"") + inFName + "\" \"" + outFName + "\"";

    try {
        system(cmd.c_str());
    }
    catch (...) {
        std::cerr << "CopyFile failed: " << cmd << std::endl;
    }
}
