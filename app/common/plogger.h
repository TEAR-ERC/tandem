#ifndef PLOGGER_20240808_H
#define PLOGGER_20240808_H

#include <iostream>
#include <sstream>
#include <array>
#include <mpi.h>
#include <unistd.h>
#include <sys/utsname.h>
#include <time.h>

namespace tndm {

class ParallelLogBase
{
public:
  MPI_Comm    comm_;
  int         commsize_, commrank_;
  int         log_level_;
  std::string filename_;
  int         file_exists_;
  FILE        *fp_;
  std::string creation_timestamp_;

   ParallelLogBase(MPI_Comm comm);
  ~ParallelLogBase(void);

  virtual void printf(const char * format, ...) {}
  virtual void print_carray_d(const char name[], size_t len, const double field[]) {}
};

class ParallelLog : public ParallelLogBase
{
private:
  void open(void);
  void close(void);

public:
  ParallelLog(MPI_Comm);
  ~ParallelLog(void);

  void printf(const char * format, ...);

  void print_carray_d(const char name[], size_t len, const double _arr[])
  {
    if (this->file_exists_ == 0) { open(); }
    fprintf(this->fp_, "%s (double): {\n  ", name);
    for (size_t t=0; t<len; t++) {
      fprintf(this->fp_,"%lf, ",_arr[t]);
    }
    fprintf(this->fp_,"\n}\n");
    fflush(this->fp_);
  }

};

} // namespace tndm

#endif // PLOGGER_20240808_H
