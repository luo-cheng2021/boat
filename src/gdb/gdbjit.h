#ifndef GDBJIT_H_
#define GDBJIT_H_

#include <cstdio>
#include <cstdint>
#include <vector>

class GDBJit{
	struct debugInfo{
		size_t offset;
		const char *file;
		int line;

		debugInfo(size_t offset, const char *file, int line) : offset(offset), file(file), line(line) {}
	};
	std::vector<debugInfo> debugEntries;

public:
	// add source line information
	void addDebugLine(size_t offset, const char *file_name=__builtin_FILE(), int line_number=__builtin_LINE());

	// register code with GDB with function name and function address + size
	void addCodeSegment(const char *name, uint64_t addr, uint64_t size);
};

#endif
