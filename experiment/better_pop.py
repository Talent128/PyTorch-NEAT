from __future__ import print_function
from neat.population import Population, CompleteExtinctionException
from neat.six_util import iteritems, itervalues

try:
    import cPickle as pickle  # pylint: disable=import-error
except ImportError:
    import pickle  # pylint: disable=import-error


class better_Population(Population):
    def __init__(self, config, initial_state=None):
        super().__init__(config, initial_state)

    def run(self, fitness_function, n=None, reeval_first_gen=False):
        """
        运行 NEAT 的遗传算法,最多进行 `n` 代进化。
        重新调整 `Population` 类的 `run()` 方法,使得在生成新一代种群之前先保存当前种群数据。

        @param reeval_first_gen: 是否在第一代时,将所有个体的适应度(fitness)设为 `None`,以强制重新评估。
        """
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")
        if reeval_first_gen:                #源码改动1：新增参数。作用：在第一代强制把所有个体 fitness 设置为 None，迫使系统对第一代重新评估
            for g in self.population:
                self.population[g].fitness = None

        k = 0       # 记录当前代的编号
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation) # 记录当前世代开始

            # 计算所有个体的适应度(用户提供的适应度评估函数)
            fitness_function(list(iteritems(self.population)), self.config)

            # 收集和报告统计信息。
            best = None     # 统计当前代适应度最高的个体
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            # 记录当前代的统计信息
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # 记录有史以来最优个体
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best
            
            # 检查是否达到适应度终止条件
            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break       # 达到适应度目标,终止进化

            # reorder here, so that loading gen k will have all fitness values already loaded
            # 源码改动2（关键）：先保存种群信息,再生成新一代。 
            # 作用：很多 reporter（如 neat-checkpointer, reporter）会在 end_generation() 被调用时保存/记录数据。先end_generation再reproduce可以先保存当前代信息再生成下一代。
            # 若像源码的方式会导致导致保存的数据实际对应的是：下一代的结构 + 上一代信息（如适应度）
            self.reporters.end_generation(self.config, self.population, self.species)

            # Create the next generation from the current generation.生成下一代种群
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # 检查是否所有个体都灭绝
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.用户可选：若灭绝,则重新创建种群,否则抛出异常
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.物种分类,确保种群内部按 NEAT 规则分组
            self.species.speciate(self.config, self.population, self.generation)

            self.generation += 1        # 进入下一代

        if self.config.no_fitness_termination:      # 若达到适应度终止条件
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome # 返回适应度最高的基因
