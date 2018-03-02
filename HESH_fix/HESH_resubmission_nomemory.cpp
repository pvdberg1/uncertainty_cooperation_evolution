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

/*this is the code for running memory = 0 strategies*/

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
const int numgen		= 10000;
const int outputgen		= 100;
const double numint		= 10;	//number of repeated interactions individuals have with separate interaction partners per generation
const double numround	= 10;	//number of interaction rounds per interaction
const double error		= 0.01;	//probability to make an implementation error
const int uncertainty   = 10;  //0 is the least uncertain (beta) distrubution, 10 is uniform.

//fitness
const double bo			= 2.0;	//benefit of the cooperative action to other
const double bs			= -1.0;	//mean benefit of the cooperative action to self (noise will be added to this)
const double bs_maxdev	= 2.0;	//maximum deviation from bs
const double cost_spec  = 0.0;	//cost of being a specialist (1 - c) is multiplied with total fitness (default: cost set at 0)
const double baseline   = 100.0; //baseline fitness for the entire life of the individual 

//mutation
const double mutprob	= 0.001;
const double mutsize	= 0.1;  //standard deviation of normal dist from which mutations are drawn

//individual & population
struct indiv
{
	int strat[3];				//0: general strat, 1: specialized strat 1, 2: specialized strat 2
	int specialist;				//0: generalist, 1: specialist
	double switchpoint;
	//fitness
	double w;
} 

pop[popsize];

//output
ofstream output;
ofstream output2;
ifstream input;

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
			(pop + i)->strat[j] = Random(mt);
		}
		
		(pop + i)->switchpoint = 0;
		(pop + i)->w = baseline; //individuals start with a baseline fitness!
		(pop + i)->specialist = Random(mt);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////update fitness and memory after an interaction
void update(int ind, int self, int other, double thisbs)
{
	(pop + ind)->w += self * (thisbs) + other * bo; 
}

////////////////////////////////////////////////////////////////////////////////////////////////make a decision
int decide(int ind, double mybs)
{
	int beh = -1;
	if ((pop + ind)->specialist == 1)
	{
		if (mybs < (pop + ind)->switchpoint) beh = (pop + ind)->strat[1];
		else beh = (pop + ind)->strat[2];
	}
	else beh = (pop + ind)->strat[0];

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

				double alpha = alphamat[int(((thisbs - minrange) / range) * 10000)][uncertainty]; //get the appropriate alpha parameter from the beta distribution from file
				double beta = (alpha - 1) / ((thisbs - minrange) / range) + 2 - alpha; //to make sure that 'thisbs' is the mode of this distribution, beta has to be equal to this

				boost::math::beta_distribution<> mybeta(alpha, beta);
				mybs[j] = quantile(mybeta, Uniform(mt))*range + minrange;

				if (mybs[j] < (pop + i + j)->switchpoint) act[j] = (pop + i + j)->strat[1];
				else  act[j] = (pop + i + j)->strat[2];
			}
			else
			{
				act[j] = (pop + i + j)->strat[0];
			}
		}
		
		coop += act[0] + act[1];
		interactions++;
		
		update(i, act[0], act[1], thisbs);					//update fitness
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

		for (int j = 0; j < 3; ++j)
		{
			if (Uniform(mt) < mutprob) (newpop + i)->strat[j] = abs(1-(pop + parent)->strat[j]);
			else (newpop + i)->strat[j] = (pop + parent)->strat[j];
		}

		if (Uniform(mt) < mutprob) (newpop + i)->switchpoint = (pop + parent)->switchpoint + Normal(mt) * mutsize;
		else (newpop + i)->switchpoint = (pop + parent)->switchpoint;

		if (Uniform(mt) < mutprob) (newpop + i)->specialist = abs(1-(pop + parent)->specialist);
		else (newpop + i)->specialist = (pop + parent)->specialist;

		(newpop + i)->w = baseline;
	}

	for (int i = 0; i < popsize; ++i)
	{
		*(pop + i) = *(newpop + i);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////statistics
void statistics()
{
	double sum_switchpoint = 0;
	double sum_strat[3] = { 0,0,0 };
	double sum_specialist = 0;
	double sum_defacto_heuristic = 0; // not only keeps track of 'true' heuristics, but also adds 'de facto' heuristics (i.e. generalists with switchpoint < -3 or > 1)
	double ss_switchpoint = 0;
	double av_switchpoint = 0;
	double sd_switchpoint = 0;
	
	for (int i = 0; i < popsize; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			sum_strat[j] += (pop + i)->strat[j];
		}
		sum_switchpoint += (pop + i)->switchpoint;
		sum_specialist += (pop + i)->specialist;
		sum_defacto_heuristic += 1 - (pop + i)->specialist; // add all true heuristic decision makers to 'sum_defacto_heuristic'

		if ((pop + i)->specialist == 1 && (pop + i)->switchpoint > (bs + bs_maxdev)) ++sum_defacto_heuristic; //add all specialists that are de facto generalists to 'sum_defacto_specialist'
		if ((pop + i)->specialist == 1 && (pop + i)->switchpoint < (bs - bs_maxdev)) ++sum_defacto_heuristic;
	}

	av_switchpoint += sum_switchpoint / popsize;

	for (int i = 0; i < popsize; ++i)
	{
		ss_switchpoint += ((pop + i)->switchpoint - av_switchpoint) * ((pop + i)->switchpoint - av_switchpoint);
	}

	sd_switchpoint = sqrt(ss_switchpoint / popsize);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// output stats

	output << (1.0 * coop)/(2.0 * interactions) << "\t";

	output << (1.0 * sum_specialist) / (1.0 * popsize) << "\t";

	output << (1.0 * sum_defacto_heuristic) / (1.0 * popsize) << "\t";

	output << av_switchpoint << "\t";
	
	output <<sd_switchpoint << "\t";
	
	for (int i = 0; i < 2; ++i)
	{
		output << sum_strat[i]/(1.0 * popsize) << "\t";
	}
	output << sum_strat[2]/(1.0 * popsize) << "\n";
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////write headers
void writeheaders()
{
	output << "gen\t" << "coop\t" << "specialism\t" << "defacto_heuristics\t" << "av_switchpoint\t" << "sd_switchpoint\t";
	output << "strat0\t" << "strat1\t" << "strat2\n";
}

//////////
// MAIN //
//////////

int main()
{
	output.open("output.txt");
	input.open("input2.txt");

	for (int i = 0; i < 10000; ++i)
	{
		for (int j = 0; j < 11; ++j)
		{
			input >> alphamat[i][j];
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

	output.close();
}