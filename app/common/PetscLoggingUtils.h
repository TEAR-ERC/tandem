// PetscLoggingUtils.h
#ifndef PETSC_LOGGING_UTILS_H
#define PETSC_LOGGING_UTILS_H

#include <chrono>
#include <ctime>
#include <iomanip>
#include <petscsys.h>
#include <sstream>
#include <string>

namespace tndm {

// Function to get current date and time as a string
inline std::string get_current_date_time_string() {
    // Get the current system date and time (to log in YYYY-MM-DD HH:MM:SS.mmm format)
    std::time_t current_time = std::time(nullptr);
    std::tm* time_info = std::localtime(&current_time);

    std::ostringstream oss;
    oss << std::put_time(time_info, "%Y-%m-%d %H:%M:%S");

    // Get milliseconds
    auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count() %
                    1000;
    oss << "." << std::setw(2) << std::setfill('0') << (millisec / 10);

    return oss.str();
}

inline std::string format_time(PetscReal t) {
    // convert time in seconds to "dy dd dh dm s" format
    PetscInt years, days, hours, minutes;
    PetscReal seconds;

    // Convert total time (in seconds) into years, days, hours, minutes, and seconds
    years = (PetscInt)(t / (60.0 * 60.0 * 24.0 * 365.25));
    t -= years * 60.0 * 60.0 * 24.0 * 365.25;

    days = (PetscInt)(t / (60.0 * 60.0 * 24.0));
    t -= days * 60.0 * 60.0 * 24.0;

    hours = (PetscInt)(t / (60.0 * 60.0));
    t -= hours * 60.0 * 60.0;

    minutes = (PetscInt)(t / 60.0);
    seconds = t - minutes * 60.0;

    // Create and return the formatted string
    std::ostringstream oss;
    if (years > 0) {
        oss << years << "y ";
    }
    if (days > 0) {
        oss << days << "d ";
    }
    if (hours > 0) {
        oss << hours << "h ";
    }
    if (minutes > 0) {
        oss << minutes << "m ";
    }
    if (seconds > 0 || oss.str().empty()) {
        oss << seconds << "s";
    }

    return oss.str();
};

} // namespace tndm

#endif // PETSC_LOGGING_UTILS_H
