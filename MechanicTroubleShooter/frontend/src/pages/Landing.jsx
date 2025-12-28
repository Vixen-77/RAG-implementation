import { Link } from 'react-router-dom';
import { Bot, Zap, Shield, ArrowRight } from 'lucide-react';

const Landing = () => {
  return (
    <div className="min-h-screen pt-16">
      {/* Hero Section */}
      <section className="relative px-4 pt-20 pb-32 overflow-hidden">
        <div className="max-w-7xl mx-auto text-center relative z-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8 animate-fade-in">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
            <span className="text-sm text-gray-300">AI-Powered Support 24/7</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight">
            Customer Support, <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
              Reimagined by AI
            </span>
          </h1>
          
          <p className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto">
            Experience lightning-fast responses, intelligent problem solving, and a support agent that never sleeps.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link to="/register" className="glass-button text-lg flex items-center gap-2 group">
              Start Free Trial
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link to="/login" className="px-8 py-3 rounded-lg text-gray-300 hover:bg-white/5 transition-all text-lg font-medium">
              Live Demo
            </Link>
          </div>
        </div>
        
        {/* Background Elements */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-indigo-500/20 rounded-full blur-[100px] -z-10"></div>
        <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-purple-500/20 rounded-full blur-[100px] -z-10"></div>
      </section>

      {/* Features Grid */}
      <section className="px-4 py-20 bg-black/20">
        <div className="max-w-7xl mx-auto grid md:grid-cols-3 gap-8">
          <FeatureCard 
            icon={<Bot className="w-8 h-8 text-indigo-400" />}
            title="Instant AI Responses"
            description="Our advanced agents understand context and resolve 80% of queries instantly."
          />
          <FeatureCard 
            icon={<Zap className="w-8 h-8 text-yellow-400" />}
            title="Real-time Learning"
            description="The system adapts to your business needs and learns from every interaction."
          />
          <FeatureCard 
            icon={<Shield className="w-8 h-8 text-green-400" />}
            title="Secure & Private"
            description="Enterprise-grade security ensuring your customer data remains protected."
          />
        </div>
      </section>
    </div>
  );
};

const FeatureCard = ({ icon, title, description }) => (
  <div className="glass-card group hover:-translate-y-2">
    <div className="mb-4 p-3 bg-white/5 rounded-lg w-fit group-hover:bg-white/10 transition-colors">
      {icon}
    </div>
    <h3 className="text-xl font-semibold mb-2 text-white">{title}</h3>
    <p className="text-gray-400">{description}</p>
  </div>
);

export default Landing;
