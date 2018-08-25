#include "stdafx.h"
#include "malloc.h"
#include "string.h"
#include "stdlib.h"
#include "stdio.h"
#include <windows.h>
#include <math.h>
#include <vector>
#include <list>
using namespace std;

#define SPLIT_NUM 5
#define HASH_TABLE_LEN 400
#define HZ_NUM 4000
#define MAX_SENT_LEN 32
#define MAX_LINE 64

struct prevChar {
	char prev_1[3] = "\0";
	char prev_2[3] = "\0";
	int prev_type = 0;
};

struct PYItem {
	char py[16] = "";
	vector <string> hzVec;
	PYItem* next = NULL;
};

struct uniqueStr {
	char str[64] = "";
	int count = 0;
};

struct currentProb {
	char UniHZ[MAX_SENT_LEN] = ""; // current HZ(2)
	char BiHZ[MAX_SENT_LEN] = ""; // current HZ(4)
	int UniHZCount = 0;
	int BiHZCount = 0;
};

struct fileInfo {
	FILE* fp = NULL;
	char currStr[MAX_SENT_LEN];
	int count = 0;
	bool isEnd = 0;
	bool isMin = 0;
};

struct HZPYInfo {
	char py[MAX_SENT_LEN] = "";
	int num = 0;
};

struct HZInfo {
	char hz[MAX_SENT_LEN] = "";
	int offset = 0;
	int bi_num = 0;
	int tri_num = 0;
	int uni_num = 0;
	float uni_prob = 0.0;
	vector<HZPYInfo> pyVec;
};

struct vColEntryInfo {
	char curHZ[16] = "";
	char curPY[16] = "";
	vector<float> prob;
	vector<pair<int, int>> backpoiners;
};

struct vColInfo {
	vector<vColEntryInfo> vColumn;
};

// global variables
int gTotalChar = 0;
PYItem** gPYHT; // hash table containning all unique PINYIN
prevChar gPrevChar;
uniqueStr gUniqueStr;
fileInfo gFileInfo[5];
currentProb gCurProb;
HZInfo* gBuffIndex[HZ_NUM];
char* gBuff;
vector<vColInfo> gvLattice;

// forward declarations
void train_split();
void train_seg();
unsigned int getHash(char* str);
bool PYhash_init();
bool PYhash_add(char* py, char* hz);
PYItem* PYitem_create(char* py, char* hz);
int GetHZNo(char* psW1);
void mergeSegment();
void wordCount();
void prob();
void gbuffIndex_init();
void gBuffer_init();
void writeBuffIndexToFile();	
void readInBuffIndex();
void readInBuff();
int compareStrBi(const void * a, const void * b);
int compareStrTri(const void * a, const void * b);
void splitBiTri();
float search_bi_prob(char* str);
float search_tri_prob(char* str);
void PYhashMaker();
bool pyHT_lookup(char* py, vector<string> hzVec);
vector<string> sentSeg(char* sent);
void vLatticeCreate(vector<string> sent_seg);
void vProbInit();
void vProbAdv();
void vLatticeOutput();
void vProbExit();

int main() {
	// Corpus Preprocess
	//train_split(); // evently split training data to smaller data sets
	//train_seg(); // segmentation on training data by character
	// Note: before wordCount(), first execute run.bat to sort the segmented files
	//wordCount(); // count # of words in segmented files
	//mergeSegment(); // merge segmented files into one file
	//prob(); //probabsility calculation
	//splitBiTri();
	//gbuffIndex_init();
	//gBuffer_init();
	//writeBuffIndexToFile();
	// readInbuffIndex: stores gBuffIndex (hz, unigram prob, index into gbuff)
	// readInbuff: store gBuff (bi/tri gram prob)

	// User starts here
	readInBuffIndex();
	readInBuff();
	PYhashMaker();
	while (1) {
		vector<string> sent_seg;
		char test[256];
		printf("Please input PIN YIN:\n(Enter q to quit)\n");
		scanf("%s", test);
		if (strcmp(test, "q") == 0)
			break;
		sent_seg = sentSeg(test);
		vLatticeCreate(sent_seg);
		vProbInit();
		vProbAdv();
		vLatticeOutput();
		vProbExit();
	}
	return 0;
}

float findMax(vector<float> vec) {
	float max = -999;
	for (auto v : vec) {
		if (max < v) max = v;
	}
	return max;
}

int getPositionOfMax(vector<float> vec, float max) {
	auto distance = find(vec.begin(), vec.end(), max);
	int dis = distance - vec.begin();
	if (dis == vec.size())
		dis--;
	return dis;
}

// create viterbi lattice.
// The number of columns in the lattice is determined by the number of characters
// in the processed PINYIN string inputed by user
void vLatticeCreate(vector<string> sent_seg) {
	// create start entry (skip the zeroth column)
	vector<vColEntryInfo> vColEntry;
	vColEntryInfo* entryInfo = new vColEntryInfo;
	strcat(entryInfo->curHZ, "start");
	vColEntry.push_back(*entryInfo);
	vColInfo* colInfo = new vColInfo;
	colInfo->vColumn = vColEntry;
	gvLattice.push_back(*colInfo);

	// Save word vectors to lattice. (starting from the 1st column)
	for (int i = 0; i < sent_seg.size(); i++) {
		int hashVal = getHash((char*)sent_seg.at(i).c_str());
		PYItem* item = gPYHT[hashVal];
		while (item != NULL) {
			if (strcmp(item->py, sent_seg.at(i).c_str()) == 0) {
				vector<vColEntryInfo> vColEntry;
				for (int j = 0; j < item->hzVec.size(); j++) {
					vColEntryInfo* entryInfo = new vColEntryInfo;
					// Save HZ in hzVec and corresponding py to entryInfo
					strcpy(entryInfo->curHZ, item->hzVec.at(j).c_str());
					strcpy(entryInfo->curPY, item->py);
					vColEntry.push_back(*entryInfo);
				}
				vColInfo* colInfo = new vColInfo;
				colInfo->vColumn = vColEntry;
				gvLattice.push_back(*colInfo);
				break;
			}
			item = item->next;
		}
	}
}

// calculate P(py | HZ) using MLE
float PYHZProb(int hzNo, char* py){
	if (gBuffIndex[hzNo]->pyVec.size() == 0) {
		// Apply smoothing to unmatched character
		return log(0.5);
	}
	else {
		int hzpyCount = 0; 
		int hzCount = gBuffIndex[hzNo]->uni_num;
		for (int i = 0; i < gBuffIndex[hzNo]->pyVec.size(); i++) {
			if (strcmp(gBuffIndex[hzNo]->pyVec.at(i).py, py) == 0) {
				hzpyCount = gBuffIndex[hzNo]->pyVec.at(i).num;
				break;
			}
		}
		return log(hzpyCount / hzCount);
	}
}

// Fill in the 1st and 2nd columns of the viterbi lattice
void vProbInit() {
	// add unigram prob to first column
	for (int i = 0; i < gvLattice.at(1).vColumn.size(); i++) {
		int hzNo = GetHZNo(gvLattice.at(1).vColumn.at(i).curHZ);
		// save unigram prob as first entry in the prob vec
		if (gBuffIndex[hzNo] == NULL) { // hz not in training set
			gvLattice.at(1).vColumn.at(i).prob.push_back(-999);
		}
		else {
			//float pyProb = PYHZProb(hzNo, gvLattice.at(1).vColumn.at(i).curPY);
			float hzProb = gBuffIndex[hzNo]->uni_prob;
			gvLattice.at(1).vColumn.at(i).prob.push_back(hzProb);
		}
		// update backpointers(all ponints to the start of the lattice)
		gvLattice.at(1).vColumn.at(i).backpoiners.push_back(make_pair(0, 0));
	}
	// add bigram prob to second column
	for (int i = 0; i < gvLattice.at(2).vColumn.size(); i++) {
		for (int j = 0; j < gvLattice.at(1).vColumn.size(); j++) {
			// sum up unigram from previous col and bigram from current hz	
			float probPrev = gvLattice.at(1).vColumn.at(j).prob.at(0); // unigram&py prob
			float probCur = -999;
			// keep probCur = -999 if current HZ not in training set
			if (gBuffIndex[GetHZNo(gvLattice.at(1).vColumn.at(j).curHZ)] != NULL) {
				// combine curHZ with prevHZ
				char hzBi[5] = "";
				strcat(hzBi, gvLattice.at(1).vColumn.at(j).curHZ);
				hzBi[2] = 0;
				strcat(hzBi, gvLattice.at(2).vColumn.at(i).curHZ);
				hzBi[4] = 0;
				probCur = search_bi_prob(hzBi); // bigram prob
			}
			// add prob sum to current HZ's prob vector, update prevHZ vec
			gvLattice.at(2).vColumn.at(i).prob.push_back((probPrev + probCur));
		}
		// update backpointer (points to the prev col entry with max prob)
		float max = findMax(gvLattice.at(2).vColumn.at(i).prob);
		int maxPos = getPositionOfMax(gvLattice.at(2).vColumn.at(i).prob, max);
		gvLattice.at(2).vColumn.at(i).backpoiners.push_back(make_pair(i,maxPos));
	}
}

// Fill in the viterbi lattice, starting from the 3rd column
void vProbAdv() {
	// add trigram prob and backpointers to the rest of the columns
	for (int i = 3; i < gvLattice.size(); i++) {
		// loop thro all the columns starting from the third column
		for (int j = 0; j < gvLattice.at(i).vColumn.size(); j++) {
			// loop thro all the col entries in i column
			for (int k = 0; k < gvLattice.at(i - 1).vColumn.size(); k++) {
				// loop thro all the col entries in (i-1) column
				vector<float> probTmp;
				for (int l = 0; l < gvLattice.at(i - 2).vColumn.size(); l++) {
					// loop thro all the col entries in the (i-2) column
					float probTri = -999;
					// keep probCur = -999 if current HZ not in training set
					if (gBuffIndex[GetHZNo(gvLattice.at(i - 2).vColumn.at(l).curHZ)] != NULL) {
						char hzTri[7] = "";
						strcat(hzTri, gvLattice.at(i - 2).vColumn.at(l).curHZ);
						hzTri[2] = 0;
						strcat(hzTri, gvLattice.at(i - 1).vColumn.at(k).curHZ);
						hzTri[4] = 0;
						strcat(hzTri, gvLattice.at(i).vColumn.at(j).curHZ);
						hzTri[6] = 0;
						probTri = search_tri_prob(hzTri); // trigram prob
						//int hzNo = GetHZNo(gvLattice.at(i).vColumn.at(j).curHZ);
						//float pyProb = PYHZProb(hzNo, gvLattice.at(i).vColumn.at(j).curPY);
						//probTri = pyProb + hzProb;

					}
					// save trigram probability to a temporary vector
					probTmp.push_back(probTri + gvLattice.at(i - 1).vColumn.at(k).prob.at(l));
				}
				float max = findMax(probTmp);
				int maxPos = getPositionOfMax(probTmp, max);
				gvLattice.at(i).vColumn.at(j).prob.push_back(max); // add max prob to curHZ prob vector
				gvLattice.at(i).vColumn.at(j).backpoiners.push_back(make_pair(k, maxPos));
			}
		}
	}
}

// Clear viterbi lattice
void vProbExit() {
	for (int i = 0; i < gvLattice.size(); i++) {
		// loop thro all the HZ columns in the Lattice
		for (int j = 0; j < gvLattice.at(i).vColumn.size(); j++) {
			// loop thro all the col entries in i column
			gvLattice.at(i).vColumn.at(j).backpoiners.clear();
			gvLattice.at(i).vColumn.at(j).prob.clear();
			strcpy(gvLattice.at(i).vColumn.at(j).curHZ, "");
		}
		gvLattice.at(i).vColumn.clear();
	}
	gvLattice.clear();
}

void vLatticeOutput() {
	vector<string> outputVec;
	// find the highest prob in the last column
	vector<pair<int, int>> tmpMax;
	int lastIdx = gvLattice.size() - 1;
	for (int i = 0; i < gvLattice.at(lastIdx).vColumn.size(); i++) {
		float max = findMax(gvLattice.at(lastIdx).vColumn.at(i).prob);
		int maxPos = getPositionOfMax(gvLattice.at(lastIdx).vColumn.at(i).prob, max);
		tmpMax.push_back(make_pair(max, maxPos));
	}
	float max = -999;
	int maxVecPos = 0;
	int maxColEntry = 0;
	for (int i = 0; i < tmpMax.size(); i++) {
		if (max < tmpMax.at(i).first)
			max = tmpMax.at(i).first;
	}
	for (int i = 0; i < tmpMax.size(); i++) {
		if (tmpMax.at(i).first == max) {
			maxVecPos = tmpMax.at(i).second;
			maxColEntry = i;
		}
	}

	int LatticeIdx = gvLattice.size() - 1;

	// trace back from the last backpointer
	while (LatticeIdx != 0) {
		char curHZ[16] = "";
		strcpy(curHZ, gvLattice.at(LatticeIdx).vColumn.at(maxColEntry).curHZ);
		outputVec.push_back(curHZ);
		if (LatticeIdx <= 2) {
			maxColEntry = gvLattice.at(LatticeIdx).vColumn.at(maxColEntry).backpoiners.at(0).second;
		}
		else {
			int maxVecPosTmp = gvLattice.at(LatticeIdx).vColumn.at(maxColEntry).backpoiners.at(maxVecPos).second;
			maxColEntry = gvLattice.at(LatticeIdx).vColumn.at(maxColEntry).backpoiners.at(maxVecPos).first;
			maxVecPos = maxVecPosTmp;
		}
		LatticeIdx--;
	}

	// output final results
	FILE* fout = fopen("PINYIN_Output.txt", "wb");
	for (int i = 0; i < outputVec.size(); i++) {
		printf("%s ", outputVec.at(outputVec.size() - 1 - i).c_str());
	}
	printf("\n");
	fclose(fout);
}

vector<string> sentSeg(char* sent) {
	// push each letter to queue
	vector<string> sent_seg;
	vector<string> hzVec;
	list<char> sentList;
	char* sentPtr = sent;
	bool isMatch = 0;
	char curPY[128] = "";
	strcpy(curPY, sent);
	while (strlen(curPY) != 0) {
		if (pyHT_lookup(curPY, hzVec)) {
			// Find a match. Push curPY to return vec
			sent_seg.push_back(curPY);
			// clear curPY, then
			// pop everything in sentList back to curPY
			strcpy(curPY, ""); // set curPY back to empty
			int sentListSize = sentList.size();
			for (int i = 0; i < sentListSize; i++) {
				int len = strlen(curPY);
				strcat(curPY, &sentList.front());
				curPY[len + 1] = 0;
				sentList.pop_front();
			}
		}
		else {
			// remove last letter from curPY, then
			// push last letter to the front of sentList
			int len = strlen(curPY);
			char* curPYTmp = curPY;
			sentList.push_front((*(curPYTmp + (len - 1)))); // push last letter back to the list
			char tmp[64] = "";
			strncat(curPY, curPY, len - 1); // keep curPY - 1
			curPY[len - 1] = 0;
		}
	}
	return sent_seg;
}

bool pyHT_lookup(char* py, vector<string> hzVec) {
	unsigned int hashVal = getHash(py);
	PYItem* item = gPYHT[hashVal];
	while (item != NULL) {
		if (strcmp(item->py, py) == 0) {
			hzVec = item->hzVec;
			return true;
		}
		item = item->next;
	}
	return false;
}

float search_bi_prob(char* str) {
	// str: input HZ (eg. ���)
	// prob: output prob
	char* pItem;
	char key[5] = "";
	strcpy(key, str);
	int indexToBuffIndex = GetHZNo(key);
	if (indexToBuffIndex == HZ_NUM)
		return -999; // HZ exceed HZ_NUM, return error code
	char* gBuffTemp = gBuff;
	gBuffTemp += gBuffIndex[indexToBuffIndex]->offset; // move ptr to offset
	int eleNum = gBuffIndex[indexToBuffIndex]->bi_num;
	pItem = (char*)bsearch((key + 2), gBuffTemp, eleNum, sizeof(float) + 2, compareStrBi);
	if (pItem == NULL) {
		// HZ not found, return unigram prob * 10
		int indexToSecondHZ = GetHZNo(key + 2);
		float uniFirst = gBuffIndex[indexToBuffIndex]->uni_prob;
		float uniSecond = gBuffIndex[indexToSecondHZ]->uni_prob;
		return ((uniFirst + uniSecond) * 10);
	}
	char hz[3] = "";
	memcpy(hz, pItem, 2);
	hz[2] = 0;
	float prob = 0.0;
	memcpy(&prob, pItem + 2, sizeof(float));
	return prob;
}

float search_tri_prob(char* str) {
	// str: input HZ (eg. ������)
	// prob: output prob
	char* pItem;
	char key[7] = "";
	strcpy(key, str);
	int indexToBuffIndex = GetHZNo(key);
	char* gBuffTemp = gBuff;
	gBuffTemp += gBuffIndex[indexToBuffIndex]->offset; // move ptr to offset
	int skipNum = gBuffIndex[indexToBuffIndex]->bi_num;
	gBuffTemp += (skipNum * (sizeof(float) + 2));
	int eleNum = gBuffIndex[indexToBuffIndex]->tri_num;
	pItem = (char*)bsearch((key + 2), gBuffTemp, eleNum, sizeof(float) + 4, compareStrTri);
	if (pItem == NULL) {
		// HZ not found, return bigram prob * 10
		char firstBi[5] = "";
		strncpy(firstBi, str, 4);
		firstBi[4] = 0;
		float biFirst = search_bi_prob(firstBi);
		float biSecond = search_bi_prob(str+2);
		return ((biFirst + biSecond)*10);
	}
	char hz[5] = "";
	memcpy(hz, pItem, 4);
	hz[4] = 0;
	float prob = 0.0;
	memcpy(&prob, pItem + 4, sizeof(float));
	return prob;
}

// evently split training data to smaller data sets
void train_split() {
	FILE *fp = fopen("corpus.txt", "rb");
	// get file size
	fseek(fp, 0, SEEK_END);
	int flen = ftell(fp);
	rewind(fp);
	// read file
	char* buff = (char *)malloc(flen + 1);
	fread(buff, sizeof(char), flen, (FILE*)fp);

	// get # of /n in training data
	char* ptr = buff;
	char tar[3];
	int i = 0;
	int count = 0;
	while (i < flen) {
		if ((*ptr & 0x80) != 0x80) {
			strncpy(tar, ptr, 1);
			tar[1] = '\0';
			if (strcmp(tar, "\n") == 0) {
				count++;
			}
			ptr += 1;
			i += 1;
		}
		else {
			ptr += 2;
			i += 2;
		}
	}

	// write to smaller files, use /n as splitting point 
	int count_small = count / SPLIT_NUM;
	char file_name[64] = "";
	rewind(fp);
	ptr = buff;
	char* ptr_ori = buff;
	for (int i = 0; i < SPLIT_NUM; i++) {
		sprintf(file_name, "train-%d.txt", i + 1);
		FILE* fpw = fopen(file_name, "ab+");
		char tar[3];
		int j = 0;
		int count_n = 0;
		while (j < flen) {
			if ((*ptr & 0x80) == 0x80) {
				strncpy(tar, ptr, 2);
				tar[2] = '\0';
				fwrite(tar, 1, 2, fpw);
				ptr += 2;
				j += 2;
			}
			else {
				strncpy(tar, ptr, 1);
				tar[1] = '\0';
				ptr += 1;
				j += 1;
				if (strcmp(tar, "\n") == 0) {
					count_n++;
					if (count_n == count_small)
						break;
				}
				fwrite(tar, 1, 1, fpw);
			}
		}
		fclose(fpw);
	}
	fclose(fp);
	free(buff);
}

void train_seg() {
	char fn_read[1024];
	char fn_write[1024];
	for (int i = 0; i < SPLIT_NUM; i++) {
		sprintf(fn_read, "train-%d.txt", i + 1);
		FILE* fp = fopen(fn_read, "rb");
		// get file size
		fseek(fp, 0, SEEK_END);
		int flen = ftell(fp);
		rewind(fp);
		// read file
		char* buff = (char *)malloc(flen + 1);
		fread(buff, sizeof(char), flen, (FILE*)fp);
		char* ptr = buff;
		char tar[3];
		char tar_bi[5];
		char tar_tri[7];
		int j = 0;
		int trigram_clock = 0;
		sprintf(fn_write, "trainSeg-%d.txt", i + 1);
		fp = fopen(fn_write, "wb");

		while (j < flen) {
			if ((*ptr & 0x80) == 0x80) {
				strncpy(tar, ptr, 2);
				tar[2] = '\0';
				fprintf(fp, "%s\n", tar);
				if (strcmp(gPrevChar.prev_1, "\0") != 0) {
					strcpy(tar_bi, gPrevChar.prev_1);
					strncat(tar_bi, ptr, 2);
					tar_bi[4] = '\0';
					fprintf(fp, "%s\n", tar_bi);
				}
				if (strcmp(gPrevChar.prev_2, "\0") != 0) {
					strcpy(tar_tri, gPrevChar.prev_2);
					strcat(tar_tri, gPrevChar.prev_1);
					strncat(tar_tri, ptr, 2);
					tar_tri[6] = '\0';
					fprintf(fp, "%s\n", tar_tri);
				}
				// save prev_1 to prev_2
				if (trigram_clock >= 1)
					strcpy(gPrevChar.prev_2, gPrevChar.prev_1);
				// save tar to prev_1
				strncpy(gPrevChar.prev_1, ptr, 2);
				gPrevChar.prev_1[2] = '\0';
				gTotalChar++;
				ptr += 2;
				j += 2;
				trigram_clock++;
			}
			else {
				strncpy(tar, ptr, 1);
				tar[1] = '\0';
				ptr += 1;
				j += 1;
				if ((strcmp(tar, "/r") == 0) || (strcmp(tar, "/n") == 0)) {
					continue;
				}
				fprintf(fp, "%s\n", tar);
				if (strcmp(gPrevChar.prev_1, "\0") != 0) {
					strcpy(tar_bi, gPrevChar.prev_1);
					strncat(tar_bi, ptr, 2);
					tar_bi[4] = '\0';
					fprintf(fp, "%s\n", tar_bi);
				}
				if (strcmp(gPrevChar.prev_2, "\0") != 0) {
					strcpy(tar_tri, gPrevChar.prev_2);
					strcat(tar_tri, gPrevChar.prev_1);
					strncat(tar_tri, ptr, 2);
					tar_tri[6] = '\0';
					fprintf(fp, "%s\n", tar_tri);
				}
				// save prev_1 to prev_2
				if (trigram_clock >= 1)
					strcpy(gPrevChar.prev_2, gPrevChar.prev_1);
				// save tar to prev_1
				strncpy(gPrevChar.prev_1, ptr, 1);
				gPrevChar.prev_1[1] = '\0';
				gTotalChar++;
				trigram_clock++;
			}
		}
		fclose(fp);
		free(buff);
	}
}

float calculate_unigram(int strCount, int total, int totalUnique) {
	// with laplace smoothing
	float prob = (float)(strCount + 1) / (total + totalUnique);
	return log(prob); // take log
}

float calculate_bigram(int biHZCount, int uniHZCount, int totalUnique) {
	// with laplace smoothing	
	float prob = (float)(biHZCount + 1) / (uniHZCount + totalUnique);
	return log(prob); // take log
}

float calculate_trigram(int triHZCount, int biHZCount, int totalUnique) {
	// with laplace smoothing
	float prob = (float)(triHZCount + 1) / (biHZCount + totalUnique);
	return log(prob); // take log
}

// generate hash key
unsigned int getHash(char* str)
{
	unsigned int hash = 0;
	int len = strlen(str);
	for (int i = 0; i < len; i++) {
		hash += (*str);
		str++;
	}
	return (hash % HASH_TABLE_LEN);
}

// initialize char hash table
bool PYhash_init() {
	gPYHT = new PYItem*[HASH_TABLE_LEN];
	for (int i = 0; i < HASH_TABLE_LEN; i++) {
		gPYHT[i] = NULL;
	}
	return true;
}

bool PYhash_add(char* py, char* hz) {
	// ȥ����
	char pyClean[16] = "";
	int pyLen = strlen(py);
	strncpy(pyClean, py, pyLen - 1);
	pyClean[pyLen] = 0;
	// check hash table
	unsigned int hashVal = getHash(pyClean);
	if (gPYHT[hashVal] == NULL) { // create new PYItem
		gPYHT[hashVal] = PYitem_create(pyClean, hz);
	}
	else {
		PYItem* pyItem = gPYHT[hashVal];
		while (1) {
			if (strcmp(pyItem->py, pyClean) == 0) {
				for (int i = 0; i < pyItem->hzVec.size(); i++) {
					if (strcmp(hz, pyItem->hzVec.at(i).c_str()) == 0)
						return false;
				}
				pyItem->hzVec.push_back(hz);
				return true;
			}
			if (pyItem->next == NULL) {
				pyItem->next = PYitem_create(pyClean, hz);
				break;
			}
			else {
				pyItem = pyItem->next;
			}
		}
		return true;
	}
}

PYItem* PYitem_create(char* py, char* hz) {
	PYItem* item = new PYItem;
	strcpy(item->py, py);
	item->hzVec.push_back(hz);
	return item;
}

void PYhashMaker() {
	// initialize PY hash table
	PYhash_init();
	FILE* fp = fopen("PINYIN.txt", "rb");
	// get file size
	fseek(fp, 0, SEEK_END);
	int flen = ftell(fp);
	rewind(fp);
	// read file
	char* buff = (char *)malloc(flen + 1);
	fread(buff, sizeof(char), flen, (FILE*)fp);
	rewind(fp);

	while (fgets(buff, MAX_SENT_LEN, fp) != NULL) {
		if (buff[strlen(buff) - 1] == '\n') {
			buff[strlen(buff) - 1] = 0;
		}
		if (buff[strlen(buff) - 1] == '\r') {
			buff[strlen(buff) - 1] = 0;
		}

		// get HZ
		char curHZ[3] = "";
		char* token = strtok(buff, " ");
		if (token == NULL)
			continue;
		int HZNo = GetHZNo(token);
		if (HZNo > HZ_NUM || HZNo < 0) // only keep HZ under HZ_NUM
			continue;
		strcpy(curHZ, token);
		token = strtok(NULL, " ");
		while (token != NULL) {
			PYhash_add(token, curHZ);
			token = strtok(NULL, " ");
		}
	}
}

void getLine(fileInfo *fI) {
	char tmp[MAX_SENT_LEN];
	if (fgets(tmp, MAX_SENT_LEN, fI->fp) != NULL) {
		if (tmp[strlen(tmp) - 1] == '\n') {
			tmp[strlen(tmp) - 1] = 0;
		}
		if (tmp[strlen(tmp) - 1] == '\r') {
			tmp[strlen(tmp) - 1] = 0;
		}

		char* token = strtok(tmp, " ");
		strcpy(fI->currStr, token);
		token = strtok(NULL, " ");
		fI->count = atoi(token);
	}
	else {
		strcpy(fI->currStr, "end");
		fI->isEnd = 1;
	}
}

void mergeInit() {
	char fn_read[1024];
	for (int i = 0; i < SPLIT_NUM; i++) {
		sprintf(fn_read, "trainSegSortedc-%d.txt", i + 1);
		FILE* fp = fopen(fn_read, "rb");
		gFileInfo[i].fp = fp;
	}
	for (int j = 0; j < SPLIT_NUM; j++) {
		getLine(&gFileInfo[j]);
	}
}

bool terminateAllFiles() {
	int isEnd = 0;
	for (int i = 0; i < SPLIT_NUM; i++) {
		if (gFileInfo[i].isEnd)
			isEnd++;
	}
	if (isEnd == SPLIT_NUM)
		return true;
	return false;
}

int cmp(const void *a, const void *b) {
	fileInfo* tmpa = (fileInfo*)a;
	fileInfo* tmpb = (fileInfo*)b;
	return strcmp(tmpa->currStr, tmpb->currStr);
}

void CreateArr(char** arr, int *len) {
	for (int i = 0; i < SPLIT_NUM; i++) {
		if (gFileInfo[i].isEnd) {
			strcpy(arr[i], "");
			continue;
		}
		arr[i] = new char[MAX_SENT_LEN];
		strcpy(arr[i], gFileInfo[i].currStr);
		len++;
	}
}
	
void GetLen(int *len) {
	for (int i = 0; i < SPLIT_NUM; i++) {
		if (!gFileInfo[i].isEnd)
			len++;
	}
}

void mergeSegment() {
	FILE *fout = fopen("trainSegMerged.txt", "wb");
	mergeInit();

	while (!terminateAllFiles()) {
		int len = 5;
		int totalCnt = 0;
		int uniqueCnt = 0;
		bool counter = 1;
		char* arr[5];
		qsort(gFileInfo, len, sizeof(fileInfo), cmp);

		for (int i = 0; i < len; i++) {
			if (gFileInfo[i].isEnd)
				continue;
			if (counter) {
				uniqueCnt += gFileInfo[i].count;
				gFileInfo[i].isMin = 1;
				counter = 0;
			}
			if (((i + 1)<len) && strcmp(gFileInfo[i].currStr, gFileInfo[i + 1].currStr) == 0) {
				gFileInfo[i + 1].isMin = 1;
				uniqueCnt += gFileInfo[i + 1].count;
			}
			else {
				break;
			}
		}
		totalCnt += uniqueCnt;
		// write to file
		if (gFileInfo[0].currStr[0] == 0) {
			continue;
		}
		if (strlen(gFileInfo[0].currStr) == 2) {
			fprintf(fout, "%s %d %d\n", gFileInfo[0].currStr, uniqueCnt, 1);
		}
		else if (strlen(gFileInfo[0].currStr) == 4) {
			fprintf(fout, "%s %d %d\n", gFileInfo[0].currStr, uniqueCnt, 2);
		}
		else if (strlen(gFileInfo[0].currStr) == 6) {
			fprintf(fout, "%s %d %d\n", gFileInfo[0].currStr, uniqueCnt, 3);
		}
		// update pointer
		for (int i = 0; i < len; i++) {
			if (gFileInfo[i].isMin) {
				getLine(&gFileInfo[i]);
				gFileInfo[i].isMin = 0;
			}
		}
	}
	fclose(fout);
}

void wordCount() {
	char fn_read[1024];
	char fn_write[1024];
	for (int i = 0; i < SPLIT_NUM; i++) {
		sprintf(fn_read, "trainSegSorted-%d.txt", i + 1);
		sprintf(fn_write, "trainSegSortedc-%d.txt", i + 1);
		FILE* fp = fopen(fn_read, "rb");
		FILE* fp_out = fopen(fn_write, "wb");
		// get file size
		fseek(fp, 0, SEEK_END);
		int flen = ftell(fp);
		rewind(fp);
		// read file
		char* buff = (char *)malloc(flen + 1);
		fread(buff, sizeof(char), flen, (FILE*)fp);
		rewind(fp);
		char prev[64] = "";

		while (fgets(buff, MAX_SENT_LEN, fp) != NULL) {
			if (buff[strlen(buff) - 1] == '\n') {
				buff[strlen(buff) - 1] = 0;
			}
			if (buff[strlen(buff) - 1] == '\r') {
				buff[strlen(buff) - 1] = 0;
			}
			if (strcmp(buff, "") == 0)
				continue;
			if (strcmp(buff, gUniqueStr.str) != 0) { // next word
				if (strcmp(gUniqueStr.str, "") != 0) { // skip the first str
					fprintf(fp_out, "%s %d\n", gUniqueStr.str, gUniqueStr.count);
				}
				strcpy(gUniqueStr.str, buff);
				gUniqueStr.count = 1;
			}
			else { // same word
				gUniqueStr.count++;
			}
		}
		strcpy(gUniqueStr.str, "");
		gUniqueStr.count = 0;
		fclose(fp);
		fclose(fp_out);
		delete(buff);
	}
}

// calculate unigram, bigram, trigram prob of merged training file
void prob() {
	FILE* fp = fopen("trainSegMerged.txt", "rb");
	FILE* fp_out = fopen("trainSegMergedProb.txt", "wb");
	char szLine[MAX_SENT_LEN] = "";
	char curStr[MAX_SENT_LEN] = "";
	int totalcount = 0;
	int totalUniqueUni, totalUniqueBi, totalUniqueTri;
	totalUniqueUni = totalUniqueBi = totalUniqueTri = 0;
	while (fgets(szLine, MAX_SENT_LEN, fp) != NULL) {
		char* token = strtok(szLine, " ");
		token = strtok(NULL, " "); // get count
		totalcount += atoi(token);
		token = strtok(NULL, " "); // get type (uni/bi/tri)
		if (atoi(token) == 1)
			totalUniqueUni++;
		else if (atoi(token) == 2)
			totalUniqueBi++;
		else if (atoi(token) == 3)
			totalUniqueTri++;
	}
	rewind(fp);
	while (fgets(szLine, MAX_SENT_LEN, fp) != NULL) {
		char* token = strtok(szLine, " "); // get HZ
		strcpy(curStr, token);
		int HZNo = GetHZNo(curStr);
		if (HZNo < 0 || HZNo > HZ_NUM)
			continue;
		token = strtok(NULL, " "); // get HZ count
		int curStrCount = atoi(token);
		if (strlen(curStr) == 2) { // UniHZ
			strcpy(gCurProb.UniHZ, curStr);
			gCurProb.UniHZCount = curStrCount;
			//calculate Unigram
			float unigram = calculate_unigram(curStrCount, totalcount, totalUniqueUni);
			// write Unigram to file
			fprintf(fp_out, "%s %d %.6f %d\n", curStr, 1, unigram, curStrCount);
		}
		if (strlen(curStr) == 4) {
			strcpy(gCurProb.BiHZ, curStr);
			gCurProb.BiHZCount = curStrCount;
			// calculate Bigram
			float bigram = calculate_bigram(curStrCount, gCurProb.UniHZCount, totalUniqueBi);
			// write bigram to file
			//fprintf(fp_out, "%s %d %.16f\n", curStr, curStrCount, bigram);
			fprintf(fp_out, "%s %d %.6f\n", curStr, 2, bigram);
		}
		if (strlen(curStr) == 6) {
			// calculate Trigram
			float trigram = calculate_trigram(curStrCount, gCurProb.BiHZCount, totalUniqueTri);
			//write trigram to file
			//fprintf(fp_out, "%s %d %.16f\n", curStr, curStrCount, trigram);
			fprintf(fp_out, "%s %d %.6f\n", curStr, 3, trigram);
		}
	}
	fclose(fp);
	fclose(fp_out);
}

void pyVecInit() {
	FILE* fp = fopen("PYCorpus.txt", "rb");
	char line[MAX_LINE];
	char prevHZ[3] = "";
	char prevPY[16] = "";
	int pyCount = 0;
	while (fgets(line, MAX_LINE, fp) != NULL) {
		if (line[strlen(line) - 1] == '\n') {
			line[strlen(line) - 1] = 0;
		}
		if (line[strlen(line) - 1] == '\r') {
			line[strlen(line) - 1] = 0;
		}
		char curHZ[3] = "";
		char* token = strtok(line, " "); // get hz
		strcpy(curHZ, token);
		token = strtok(NULL, " "); // get py
		char pyClean[16] = "";
		int pyLen = strlen(token);
		strncpy(pyClean, token, pyLen - 1);
		pyClean[pyLen] = 0;
		token = strtok(NULL, " "); // get count
		if (strcmp(curHZ, prevHZ) == 0 && strcmp(pyClean, prevPY) == 0) {
			pyCount += atoi(token);
		}
		else {
			int curHZNo = GetHZNo(curHZ);
			if (gBuffIndex[curHZNo] != NULL) {
				HZPYInfo* item = new HZPYInfo;
				strcpy(item->py, pyClean);
				item->num = pyCount;
				gBuffIndex[curHZNo]->pyVec.push_back(*item);
				strcpy(prevHZ, curHZ);
				strcpy(prevPY, pyClean);
				pyCount = 0;
			}
		}

	}

}

// create viterbi lattices for bigram and trigram
void gbuffIndex_init() {
	FILE *fp = fopen("trainSegProb.txt", "rb");
	char line[MAX_LINE];
	char curHZ[MAX_SENT_LEN] = "";
	float curUniProb = 0.0;
	int curBiNum = 0;
	int curTriNum = 0;
	int curUniCount = 0;
	while (fgets(line, MAX_LINE, fp) != NULL) {
		if (line[strlen(line) - 1] == '\n') {
			line[strlen(line) - 1] = 0;
		}
		if (line[strlen(line) - 1] == '\r') {
			line[strlen(line) - 1] = 0;
		}
		char* token = strtok(line, " ");
		if ((token[0] & 0x80) != 0x80) { // skip nonHZ
			continue;
		}
		int curHZNo = GetHZNo(token);
		int biHZNo = 0;
		int triHZNo = 0;
		if (strlen(token) > 2) {
			biHZNo = GetHZNo(token + 2);
			if (strlen(token) == 6)
				triHZNo = GetHZNo(token + 4);
		}
		if (curHZNo < 0 || curHZNo >= HZ_NUM || biHZNo < 0 || biHZNo >= HZ_NUM
			|| triHZNo < 0 || triHZNo >= HZ_NUM) // keep HZ under HZ_NUM
			continue;
		if (strcmp(curHZ, "") == 0) { // save first HZ to curHZ
			strncpy(curHZ, token, 2);
			curHZ[3] = 0;
		}
		if (strcmp(curHZ, "") != 0 && curHZNo != GetHZNo(curHZ)) { // next HZ
																   //update_HZInfo
			int currHZNo = GetHZNo(curHZ);
			if (gBuffIndex[currHZNo] != NULL) {
				strncpy(curHZ, token, 2); // update curHZ
				curHZ[3] = 0;
				curBiNum = curTriNum = curUniProb = 0;
				continue;
			}
			gBuffIndex[currHZNo] = new HZInfo;
			strcpy(gBuffIndex[currHZNo]->hz, curHZ);
			gBuffIndex[currHZNo]->uni_num = curUniCount;
			gBuffIndex[currHZNo]->bi_num = curBiNum;
			gBuffIndex[currHZNo]->tri_num = curTriNum;
			gBuffIndex[currHZNo]->uni_prob = curUniProb;

			strncpy(curHZ, token, 2); // update curHZ
			curHZ[3] = 0;
			curBiNum = curTriNum = curUniProb = 0;
		}
		if (strlen(token) == 4) { // add to bigram total
			curBiNum++;
		}
		else if (strlen(token) == 6) { // add to trigram total
			curTriNum++;
		}
		else if (strlen(token) == 2) {
			token = strtok(NULL, " "); // get prob
			curUniProb = atof(token);
			token = strtok(NULL, " "); // get count
			curUniCount = atoi(token);
		}
	}
	int curHZNo = GetHZNo(curHZ);
	gBuffIndex[curHZNo] = new HZInfo;
	strcpy(gBuffIndex[curHZNo]->hz, curHZ);
	gBuffIndex[curHZNo]->bi_num = curBiNum;
	gBuffIndex[curHZNo]->tri_num = curTriNum;
	gBuffIndex[curHZNo]->uni_prob = curUniProb;
}

void gBuffer_init() {
	int totalBiNum = 0;
	int totalTriNum = 0;
	int curOffset = 0;

	FILE *fp = fopen("trainSegProb.txt", "rb");
	FILE *fout = fopen("HZProbBuffer.txt", "wb");
	char line[MAX_LINE];
	char curHZ[MAX_SENT_LEN] = "";
	int curType = 0;
	while (fgets(line, MAX_LINE, fp) != NULL) {
		if (line[strlen(line) - 1] == '\n')
			line[strlen(line) - 1] = 0;
		if (line[strlen(line) - 1] == '\r')
			line[strlen(line) - 1] = 0;
		int lineHZNo = 0;
		int curHZNo = 0;
		int biHZNo = 0;
		int triHZNo = 0;
		int HZType = 0;

		char* token = strtok(line, " ");
		if ((token[0] & 0x80) != 0x80 || strlen(token) == 2) // skip nonHZ and single HZ
			continue;
		lineHZNo = GetHZNo(token);
		if (strlen(token) > 2) {
			biHZNo = GetHZNo(token + 2);
			if (strlen(token) == 6)
				triHZNo = GetHZNo(token + 4);
		}
		if (lineHZNo < 0 || lineHZNo >= HZ_NUM || biHZNo < 0 || biHZNo >= HZ_NUM
			|| triHZNo < 0 || triHZNo >= HZ_NUM) // keep HZ under HZ_NUM
			continue;
		curHZNo = GetHZNo(curHZ);
		if (strcmp(curHZ, "") == 0) { // save first HZ to curHZ, save offset 0
			gBuffIndex[lineHZNo]->offset = curOffset;
		}
		else if (strcmp(curHZ, "") != 0 && lineHZNo != curHZNo) { // next HZ update_HZInfo
			if (lineHZNo == (HZ_NUM - 1))
				break;
			// update gBuffInfo.offset
			gBuffIndex[lineHZNo]->offset = curOffset;
		}
		strcpy(curHZ, token); // update curHZ
		if (strlen(curHZ) == 4) { // add to bigram total
			fwrite(curHZ + 2, sizeof(char), 2, fout);
			token = strtok(NULL, " "); // get prob
			float f = atof(token);
			fwrite(&f, sizeof(float), 1, fout);
			curOffset += (2 + sizeof(float));
		}
		if (strlen(curHZ) == 6) { // add to trigram total
			fwrite(curHZ + 2, sizeof(char), 4, fout);
			token = strtok(NULL, " "); // get prob
			float f = atof(token);
			fwrite(&f, sizeof(float), 1, fout);
			curOffset += (4 + sizeof(float));
		}
	}
	fclose(fp);
	fclose(fout);
}

void writeBuffIndexToFile()	 {
	int num = 0;
	FILE* fout = fopen("HZInfoIndex.txt", "wb");
	for (int i = 0; i < HZ_NUM; i++) {
		if (gBuffIndex[i] != 0)
			num++;
	}
	fwrite(&num, sizeof(int), 1, fout);
	for (int i = 0; i < HZ_NUM; i++) {
		if (gBuffIndex[i] != 0) {;
			fwrite(&i, sizeof(int), 1, fout);
			fwrite(gBuffIndex[i]->hz, sizeof(char), 2, fout);
			fwrite(&gBuffIndex[i]->offset, sizeof(int), 1, fout);
			fwrite(&gBuffIndex[i]->uni_num, sizeof(int), 1, fout);
			fwrite(&gBuffIndex[i]->bi_num, sizeof(int), 1, fout);
			fwrite(&gBuffIndex[i]->tri_num, sizeof(int), 1, fout);
			fwrite(&gBuffIndex[i]->uni_prob, sizeof(float), 1, fout);
		}
	}
	fclose(fout);
}

void readInBuffIndex() {
	FILE* fp = fopen("HZInfoIndex.txt", "rb");
	int num = 0;
	int index = 0;
	char hz[MAX_SENT_LEN] = "";
	int offset = 0;
	int uni_num = 0;
	int bi_num = 0;
	int tri_num = 0;
	float uni_prob = 0.0;
	fread(&num, sizeof(int), 1, fp);
	for (int i = 0; i < num; i++) {
		fread(&index, sizeof(int), 1, fp);
		fread(hz, sizeof(char), 2, fp);
		fread(&offset, sizeof(int), 1, fp);
		fread(&uni_num, sizeof(int), 1, fp);
		fread(&bi_num, sizeof(int), 1, fp);
		fread(&tri_num, sizeof(int), 1, fp);
		fread(&uni_prob, sizeof(float), 1, fp);
		gBuffIndex[index] = new HZInfo;
		strcpy(gBuffIndex[index]->hz, hz);
		gBuffIndex[index]->uni_num = uni_num;
		gBuffIndex[index]->bi_num = bi_num;
		gBuffIndex[index]->tri_num = tri_num;
		gBuffIndex[index]->offset = offset;
		gBuffIndex[index]->uni_prob = uni_prob;
	}

	// read in py info
	pyVecInit();
}

void readInBuff() {
	FILE* fp = fopen("HZProbBuffer.txt", "rb");
	// get file size
	fseek(fp, 0, SEEK_END);
	int flen = ftell(fp);
	rewind(fp);
	gBuff = new char[flen + 1];
	fread(gBuff, sizeof(char), flen, fp);
}

int compareStrBi(const void * a, const void * b) {
	return memcmp(a, b, 2);
}

int compareStrTri(const void * a, const void * b) {
	return memcmp(a, b, 4);
}

void splitBiTri() {
	char* startOfBuff = gBuff;
	FILE* fp = fopen("trainSegMergedProb.txt", "rb");
	FILE* fout = fopen("trainSegProb.txt", "wb");
	char line[MAX_LINE];
	char curHZ[MAX_SENT_LEN] = "";
	int prevHZNo = 0;
	char curHZUni[3] = "";
	int HZUniCount = 0;
	float HZUniProb = 0;
	vector <string> biHZVec;
	vector <float> biProbVec;
	vector <string> triHZVec;
	vector <float> triProbVec;
	while (fgets(line, MAX_LINE, fp) != NULL) {
		if (line[strlen(line) - 1] == '\n') {
			line[strlen(line) - 1] = 0;
		}
		if (line[strlen(line) - 1] == '\r') {
			line[strlen(line) - 1] = 0;
		}
		char* token = strtok(line, " ");
		if ((token[0] & 0x80) != 0x80) { // skip nonHZ
			continue;
		}
		int curHZNo = GetHZNo(token);
		if (prevHZNo != 0 && curHZNo != prevHZNo) {
			// we are at next char
			// save single HZ to file
			fprintf(fout, "%s %.6f %d\n", curHZUni, HZUniProb, HZUniCount);
			HZUniProb = 0;
			HZUniCount = 0;
			strcpy(curHZUni, "");
			// save two vectors to file, then clear vectors 
			for (int i = 0; i < biHZVec.size(); i++) {
				//printf("%s %.6f\n", biHZVec.at(i).c_str(), biProbVec.at(i));
				fprintf(fout, "%s %.6f\n", biHZVec.at(i).c_str(), biProbVec.at(i));
			}
			for (int i = 0; i < triHZVec.size(); i++) {
				//printf("%s %.6f\n", triHZVec.at(i).c_str(), triProbVec.at(i));
				fprintf(fout, "%s %.6f\n", triHZVec.at(i).c_str(), triProbVec.at(i));
			}
			biHZVec.clear();
			biProbVec.clear();
			triHZVec.clear();
			triProbVec.clear();
		}
		// same char, determine bi or tri
		strcpy(curHZ, token);
		token = strtok(NULL, " "); // token = str type(2/3)
		int strType = atoi(token);
		token = strtok(NULL, " "); // token = prob
		float strProb = atof(token);
		if (strType == 3) {
			triHZVec.push_back(curHZ);
			triProbVec.push_back(strProb);
		}
		else if (strType == 2){
			biHZVec.push_back(curHZ);
			biProbVec.push_back(strProb);
		}
		else { // single HZ
			token = strtok(NULL, " ");
			HZUniCount = atoi(token);
			HZUniProb = strProb;
			strcpy(curHZUni, curHZ);
		}
		prevHZNo = GetHZNo(curHZ);
	}
	fclose(fp);
	fclose(fout);
}

int GetHZNo(char* psW1) {
	int nNo1;
	nNo1 = -1;
	if (((unsigned char)*psW1 >= 0xb0) && ((unsigned char)*psW1 < 0xd8) 
		&& ((unsigned char)psW1[1] > 0xa0) && ((unsigned char)psW1[1] < 0xff)) {
		if ((unsigned char)psW1[0]>215)
			nNo1 = (((unsigned char)psW1[0] - 176) * 94 + ((unsigned char)psW1[1] - 6));

		else
			nNo1 = (((unsigned char)psW1[0] - 176) * 94 + ((unsigned char)psW1[1] - 1));
	}
	return nNo1;
}
