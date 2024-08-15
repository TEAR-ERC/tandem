#include <cstdarg>
#include <iostream>
#include <sstream>
#include <array>
#include "config.h"
#include "plogger.h"

namespace tndm {

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}

ParallelLogBase::ParallelLogBase(MPI_Comm comm)
{
  comm_ = comm;
  MPI_Comm_rank(comm_, &commrank_);
  MPI_Comm_size(comm_, &commsize_);
  std::cout << "<ParallelLogBase> Constructor executed" << std::endl;
  // Define the file name: tandem_journal_rXXXX.log
  std::ostringstream oss;
  oss << "tandem_journal_r" << commrank_ << ".log";
  filename_ = oss.str();
  log_level_ = 0;
  file_exists_ = 0;
  fp_ = NULL;
  creation_timestamp_ = currentDateTime();
}

ParallelLogBase::~ParallelLogBase(void)
{
  std::cout << "<ParallelLogBase> Destructor executed\n";
}

void ParallelLog::open(void)
{
  const char *fname = filename_.c_str();

  if (file_exists_ == 1) return;
  std::cout << "opening log file: " << filename_ << std::endl;

  fp_ = fopen(fname, "w");

  fprintf(fp_, "{\n  \"TandemLog\": {\n");
  fprintf(fp_, "    \"created\"    : \"%s\",\n", creation_timestamp_.data());

  {
    const std::string timestamp = currentDateTime();
    fprintf(fp_, "    \"opened\"     : \"%s\",\n", timestamp.data());
  }

  {
    struct utsname buffer;
    int    ierr;

    ierr = uname(&buffer);
    if (ierr == 0) {
      fprintf(fp_, "    \"host_name\"  : \"%s\",\n", buffer.nodename);
      fprintf(fp_, "    \"os_name\"    : \"%s\",\n", buffer.sysname);
      fprintf(fp_, "    \"os_release\" : \"%s\",\n", buffer.release);
      fprintf(fp_, "    \"os_version\" : \"%s\",\n", buffer.version);
      fprintf(fp_, "    \"arch\"       : \"%s\",\n", buffer.machine);
    }
  }

  {
    char *user = getlogin();
    fprintf(fp_, "    \"username\"   : \"%s\",\n", user);
    //if (user != NULL) { free(user); }
  }

  {
    // VersionString provides the commit hash from running > git rev-parse --short -q HEAD
    // The entirety of VersionString comes from running > git describe
    fprintf(fp_, "    \"git_hash\"   : \"%s\",\n", VersionString);
  }

  fprintf(fp_, "    \"commsize\"   : %d,\n", commsize_);
  fprintf(fp_, "    \"commrank\"   : %d\n", commrank_); // no comma - last entry
  fprintf(fp_, "  }\n}\n");
  fflush(fp_);

  file_exists_ = 1;
}

void ParallelLog::close(void)
{
  if (file_exists_ == 1) {
    std::cout << "closing log file: " << filename_ << std::endl;
    fclose(fp_);
    fp_ = NULL;
  }
}

ParallelLog::ParallelLog(MPI_Comm comm) : ParallelLogBase(comm)
{
  std::cout << "<ParallelLog> Constructor executed " << commrank_ << std::endl;
}

ParallelLog::~ParallelLog(void)
{
  std::cout << "<ParallelLog> Destructor executed " << commrank_ << std::endl;
  close();
}

void ParallelLog::printf(const char * format, ...)
{
  va_list argp;

  if (file_exists_ == 0) {
    open();
  }
  va_start(argp, format);
  vfprintf(fp_, format, argp);
  va_end(argp);
  fflush(fp_);
}

} // namespace tndm
