#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <vector>
#include <boost/math/distributions.hpp>
using namespace std;

/*this is the code for the adjustment of the model for the resubmission of the manuscript to Nature Communications.
There are four major changes:
1. The cost of having a conditional strategy is set to 0. This may lead to 'de facto heuristic' strategies that just evolve the threshold beyond the 
extremes of the scale. I solve this by just counting the conditional strategies for which this holds as de facto heuristic.
2. Rather than adding a baseline fitness that is different for every interaction, and keeps the lowest possible payoff of any round to exactly zero, I
now just add a standard baseline fitness of 100.
3. Rather than having a variable probability of having a new round after each round, I set a fixed number of interaction rounds. 
4. The original implementation of uncertainty has been changed. Rather than drawing a real c and another (uncorrelated) value, and weighing both values
according to the uncertainty to produce a perceived c, I now draw the perceived c from a beta distribution with mode c. Of course it is important to keep the
variance of this distribution constant for any given uncertainty. This means that both parameters of the beta distribution must change for different values of
c to hold the variance of the distribution constant. I used mathematica to create a list of values of one of the two parameters (alpha) for all different
levels of uncertainty (where 0 is an extremely narrow distribution and 10 is uniform) and with high resolution for many different values of the mode (c). 
This list is now read from file and then used to create a beta distribution for each situation (beta can be calculated based on alpha if the mode and 
variance are known).

ALSO, this version fixes a bug of the first runs for the resubmission. mybs, the variable holding the individuals' perception
of bs, was coded as an int, but it should have been a double. In this version this is fixed.
*/

//random
auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
mt19937 mt(seed);
uniform_real_distribution<double> Uniform(0, 1);
normal_distribution<double> Normal(0, 1);
uniform_int_distribution<int> Random(0, 1);

/////////////
// GLOBALS //
/////////////

//general
const int popsize		= 1000;
const int numgen		= 1000;
const int outputgen		= 1;
const double numint		= 10;	//number of repeated interactions individuals have with separate interaction partners per generation
const double numround	= 10;	//CHANGE FROM FIRST MODEL: fixed number of interaction rounds per interaction
const double error		= 0.01;	//probability to make an implementation error
const int initsd		= 0;	//initial standard deviation of weights/thresholds (mean 0). If 0, all is initialized at 0.
const int uncertainty   = 10;  //CHANGE FROM FIRST MODEL: now a beta distribution. 0 is the least uncertain dist, 1 is uniform.

//fitness
const double bo			= 2.0;	//benefit of the cooperative action to other
const double bs			= -1.0;	//mean benefit of the cooperative action to self (noise will be added to this)
const double bs_maxdev	= 2.0;	//maximum deviation from bs
const double cost_spec  = 0.0;	//cost of being a specialist (1 - c) is multiplied with total fitness - CHANGE FROM FIRST MODEL: no cost (cost set at 0)
const double baseline   = 100.0; //CHANGE FROM FIRST MODEL: baseline fitness for the entire life of the individual 

//mutation
const double mutprob	= 0.001;
const double mutsize	= 0.1;  //standard deviation of normal dist from which mutations are drawn
const double mutsizefirst = 0.1;  //mutation size of firstmove locus

//individual & population
struct indiv
{
	int strat[3][4];			//0: general strat, 1: specialized strat 1, 2: specialized strat 2
	int specialist;				//0: generalist, 1: specialist
	double switchpoint;
	//fitness
	double w;
	//memory
	int prevS;					//own previous move
	int prevO;					//other's previous move
	//firstmove
	double first[3];				
} pop[popsize];

//output
ofstream output;
ofstream output2;
ifstream input;
ifstream input2;

double mutmat[16][16];
double alphamat[10000][11];

//for following cooperation frequency
int coop;
int interactions;

///////////////
// FUNCTIONS //
///////////////

///////////////////////////////////////////////////////////////////////////////////////////////////initialize the population
void init()
{
	for (int i = 0; i < popsize; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			for (int k = 0; k < 4; ++k)
			{
				(pop + i)->strat[j][k] = Random(mt);
			}
			(pop + i)->first[j] = Uniform(mt);
		}

		(pop + i)->switchpoint = 0;
		(pop + i)->w = baseline; //individuals start with a baseline fitness!
		(pop + i)->prevS = 0;
		(pop + i)->prevO = 0;
		(pop + i)->specialist = Random(mt);
	}

	//read mutation matrix
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			input >> mutmat[i][j];
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////update fitness and memory after an interaction
void update(int ind, int self, int other, double thisbs)
{
	(pop + ind)->prevS = self;														//update memory
	(pop + ind)->prevO = other;
	(pop + ind)->w += self * (thisbs) + other * bo; 
}

////////////////////////////////////////////////////////////////////////////////////////////////make a decision using the neural network
int decide(int ind, double mybs)
{
	int beh = -1;
	if ((pop + ind)->specialist == 1)
	{
		if (mybs < (pop + ind)->switchpoint) beh = (pop + ind)->strat[1][(pop + ind)->prevO * 1 + (pop + ind)->prevS * 2];
		else beh = (pop + ind)->strat[2][(pop + ind)->prevO * 1 + (pop + ind)->prevS * 2];
	}
	else { beh = (pop + ind)->strat[0][(pop + ind)->prevO * 1 + (pop + ind)->prevS * 2]; }

	if (Uniform(mt) < error)
	{
		coop += abs(1 - beh);
		return(abs(1 - beh));
	}
	else
	{
		coop += beh;
		return(beh);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////let individuals interact
void interact(double thisbs)
{
	random_shuffle(begin(pop), end(pop)); //this shuffles the entire population so we can just pair subsequent individuals
	
	for (int i = 0; i < popsize; i = i+2)
	{
		int act[2];
		double mybs[2];
		
		//determine first move of both players...
		for (int j = 0; j < 2; ++j)
		{
			if ((pop + i + j)->specialist == 1)
			{
				double minrange = bs - bs_maxdev;
				double range = bs_maxdev * 2;

				double alpha = alphamat[int(((thisbs - minrange)/range) * 10000)][uncertainty]; //get the appropriate alpha parameter from the beta distribution from file
				double beta = (alpha - 1) / ((thisbs - minrange)/range) + 2 - alpha; //to make sure that 'thisbs' is the mode of this distribution, beta has to be equal to this
				
				boost::math::beta_distribution<> mybeta(alpha, beta);
				mybs[j] = quantile(mybeta, Uniform(mt))*range + minrange;

				if (mybs[j] < (pop + i + j)->switchpoint) act[j] = (Uniform(mt) < (pop + i + j)->first[1]);
				else  act[j] = (Uniform(mt) < (pop + i + j)->first[2]);
			}
			else
			{
				act[j] = (Uniform(mt) < (pop + i + j)->first[0]);
			}
		}
		
		coop += act[0] + act[1];
		interactions++;
		
		update(i, act[0], act[1], thisbs);					//update fitness and memory
		update(i + 1, act[1], act[0], thisbs);

		for (int j = 0; j < numround; ++j)					// repeat interaction
		{
			interactions++;
			act[0] = decide(i, mybs[0]);
			act[1] = decide(i + 1, mybs[1]);
			update(i, act[0], act[1], thisbs);
			update(i + 1, act[1], act[0], thisbs);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////reproduction
void reproduce()
{
	indiv newpop[popsize];
	
	//first all individuals pay cost of specialism...
	for (int i = 0; i < popsize; ++i)
	{
		if ((pop + i)->specialist == 1) { (pop + i)->w *= 1.0 - cost_spec; };
	}
	
	double w[popsize]; 
	double sumw = 0;

	for (int i = 0; i < popsize; ++i)
	{
		if ((pop + i)->w < 1) (pop + i)->w = 1; //fitness is unlikely to end up below 1 (baseline is 100), but if it happens, it is set to 1
		w[i] = sumw + (pop + i)->w;
		sumw += (pop + i)->w;
	}

	for (int i = 0; i < popsize; ++i)
	{
		double pick = Uniform(mt) * sumw;

		int min = 0;
		int max = popsize - 1;
		int parent = -1;
		int mid = (int)((max + min)*0.5);

		while ((max - min) > 1)
		{
			if (w[mid] >= pick) max = mid;
			else min = mid;
			mid = (int)((max + min)*0.5);
		}
		parent = max;

		int oldstrat[3] = { 0,0,0 };
		
		for (int j = 0; j < 3; ++j)
		{
			int factor = 1;
			for (int k = 0; k < 4; ++k)
			{
				oldstrat[j] += (pop + parent)->strat[j][k] * factor;
				factor *= 2;
			}
		}

		//strategy mutation according to the mutation matrix mutmat...
		for (int j = 0; j < 3; ++j)
		{
			if (Uniform(mt) < mutprob)
			{
				double pick = Uniform(mt);
				int k = 0;
				while (pick > mutmat[oldstrat[j]][k])
				{
					k++;
				}

				if (k % 2 == 0) { (newpop + i)->strat[j][0] = 0; }
				else { (newpop + i)->strat[j][0] = 1; }
				if (int(floor(k / 2)) % 2 == 0) { (newpop + i)->strat[j][1] = 0; }
				else { (newpop + i)->strat[j][1] = 1; }
				if (int(floor(k / 4)) % 2 == 0) { (newpop + i)->strat[j][2] = 0; }
				else { (newpop + i)->strat[j][2] = 1; }
				if (int(floor(k / 8)) == 0) { (newpop + i)->strat[j][3] = 0; }
				else { (newpop + i)->strat[j][3] = 1; }
			}
			else
			{
				for (int k = 0; k < 4; ++k)
				{
					(newpop + i)->strat[j][k] = (pop + parent)->strat[j][k];
				}
			}

			if (Uniform(mt) < mutprob)
			{
				(newpop + i)->first[j] = (pop + parent)->first[j] + Normal(mt) * mutsizefirst;
				if ((newpop + i)->first[j] < 0.0) (newpop + i)->first[j] = 0.0;
				if ((newpop + i)->first[j] > 1.0) (newpop + i)->first[j] = 1.0;
			}
			else (newpop + i)->first[j] = (pop + parent)->first[j];
		}

		if (Uniform(mt) < mutprob) (newpop + i)->switchpoint = (pop + parent)->switchpoint + Normal(mt) * mutsize;
		else (newpop + i)->switchpoint = (pop + parent)->switchpoint;

		if (Uniform(mt) < mutprob) (newpop + i)->specialist = abs(1-(pop + parent)->specialist);
		else (newpop + i)->specialist = (pop + parent)->specialist;

		(newpop + i)->prevS = 0;
		(newpop + i)->prevO = 0;
		(newpop + i)->w = baseline;
	}

	for (int i = 0; i < popsize; ++i)
	{
		*(pop + i) = *(newpop + i);
	}
}

void isolated_contexts()
{
	output2 << "bs"	<< "\t";
	output2 << "coop" << "\n";

	for (int i = 0; i < 101; ++i)
	{
		coop = 0;
		interactions++;

		double thisbs = (bs - bs_maxdev + (i*1.0)/100.0 * 2 * bs_maxdev);

		for (int j = 0; j < 100; j++)
		{
			interact(thisbs);
		}

		output2 << thisbs << "\t";
		output2 << (1.0 * coop) / (2.0 * interactions) << "\n";
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////statistics
void statistics()
{
	double sum_switchpoint = 0;
	double sum_first[3] = { 0,0,0 };
	double sum_specialist = 0;
	double sum_defacto_heuristic = 0; // CHANGE FROM FIRST MODEL: not only keeps track of 'true' heuristics, but also adds 'de facto' heuristics (i.e. generalists with switchpoint < -3 or > 1)
	double ss_switchpoint = 0;
	double ss_first[3] = { 0,0,0 };
	double av_switchpoint = 0;
	double av_first[3] = { 0,0,0 };
	double sd_switchpoint = 0;
	double sd_first[3] = { 0,0,0 };
	
	for (int i = 0; i < popsize; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			sum_first[j] += (pop + i)->first[j];
		}
		sum_switchpoint += (pop + i)->switchpoint;
		sum_specialist += (pop + i)->specialist;
		sum_defacto_heuristic += 1 - (pop + i)->specialist; // add all true heuristic decision makers to 'sum_defacto_heuristic'

		if ((pop + i)->specialist == 1 && (pop + i)->switchpoint > (bs + bs_maxdev)) ++sum_defacto_heuristic; //add all specialists that are de facto generalists to 'sum_defacto_specialist'
		if ((pop + i)->specialist == 1 && (pop + i)->switchpoint < (bs - bs_maxdev)) ++sum_defacto_heuristic;
	}

	for (int j = 0; j < 3; ++j)
	{
		av_first[j] += sum_first[j] / popsize;
	}
	av_switchpoint += sum_switchpoint / popsize;

	for (int i = 0; i < popsize; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			ss_first[j] += ((pop + i)->first[j] - av_first[j]) * ((pop + i)->first[j] - av_first[j]);
		}
		ss_switchpoint += ((pop + i)->switchpoint - av_switchpoint) * ((pop + i)->switchpoint - av_switchpoint);
	}

	for (int j = 0; j < 3; ++j)
	{
		sd_first[j] = sqrt(ss_first[j] / popsize);
	}
	sd_switchpoint = sqrt(ss_switchpoint / popsize);
	
	double stratvec[3][16];

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 16; ++j) { stratvec[i][j] = 0; }
	}
		
	for (int i = 0; i < popsize; ++i)
	{
		int thisstrat[3];

		for (int j = 0; j < 3; ++j)
		{
			thisstrat[j] = (pop + i)->strat[j][0] + 2 * (pop + i)->strat[j][1] + 4 * (pop + i)->strat[j][2] + 8 * (pop + i)->strat[j][3];
			stratvec[j][thisstrat[j]] += 1.0 / (1.0 * popsize);
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// output stats

	output << (1.0 * coop)/(2.0 * interactions) << "\t";

	output << (1.0 * sum_specialist) / (1.0 * popsize) << "\t";

	output << (1.0 * sum_defacto_heuristic) / (1.0 * popsize) << "\t";

	for (int i = 0; i < 3; ++i)
	{
		output << av_first[i] << "\t";
	}
		
	output << av_switchpoint << "\t";
	
	for (int i = 0; i < 3; ++i)
	{
		output << sd_first[i] << "\t";
	}
	
	output <<sd_switchpoint << "\t";
	
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			output << stratvec[i][j] << "\t";
		}
	}
	for (int j = 0; j < 15; ++j)
	{
		output << stratvec[2][j] << "\t";
	}
	output << stratvec[2][15] << "\n";
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////write headers
void writeheaders()
{
	output << "gen\t" << "coop\t" << "specialism\t" << "defacto_heuristics\t" << "av_first1\t" << "av_first2\t" << "av_first3\t" << "av_switchpoint\t" 
									<< "sd_first1\t" << "sd_first2\t" << "sd_first3\t" << "sd_switchpoint\t";

	output << "1_0000\t" << "1_0001\t" << "1_0010\t" << "1_0011\t" << "1_0100\t" << "1_0101\t" << "1_0110\t" << "1_0111\t";
	output << "1_1000\t" << "1_1001\t" << "1_1010\t" << "1_1011\t" << "1_1100\t" << "1_1101\t" << "1_1110\t" << "1_1111\t";
	output << "2_0000\t" << "2_0001\t" << "2_0010\t" << "2_0011\t" << "2_0100\t" << "2_0101\t" << "2_0110\t" << "2_0111\t";
	output << "2_1000\t" << "2_1001\t" << "2_1010\t" << "2_1011\t" << "2_1100\t" << "2_1101\t" << "2_1110\t" << "2_1111\t";
	output << "3_0000\t" << "3_0001\t" << "3_0010\t" << "3_0011\t" << "3_0100\t" << "3_0101\t" << "3_0110\t" << "3_0111\t";
	output << "3_1000\t" << "3_1001\t" << "3_1010\t" << "3_1011\t" << "3_1100\t" << "3_1101\t" << "3_1110\t" << "3_1111\n";
}

//////////
// MAIN //
//////////

int main()
{
	output.open("output.txt");
	output2.open("output2.txt");
	input.open("input.txt");
	input2.open("input2.txt");

	for (int i = 0; i < 10000; ++i)
	{
		for (int j = 0; j < 11; ++j)
		{
			input2 >> alphamat[i][j];
		}
	}

	writeheaders();
	
	init();
	
	for (int i = 0; i < numgen; ++i)
	{
		//if (i % 100 == 0) cout << i << "\n";
		
		coop = 0;
		interactions = 0;

		for (int j = 0; j < numint; ++j)
		{
			double thisbs = bs + (Uniform(mt) * bs_maxdev * 2.0 - bs_maxdev); //determine current game
			interact(thisbs);
		}

		if (i%outputgen == 0)
		{
			output << i << "\t";
			statistics();
		}

		reproduce();				
	}

	//isolated_contexts();

	output.close();
}