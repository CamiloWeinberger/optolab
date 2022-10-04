from pyramidal.generator_tf import Generator
import unittest
import os

class Test_Generator(unittest.TestCase):
  def test_generator(self):
    names_normalize = ['mean_std', 'mean_std_min_max', 'min_max']
    #remove npy files before test (if exist) to avoid errors in folder parameters
    for name in names_normalize:
      if os.path.exists(f'parameters/{name}_head.npy'):
        os.remove(f'parameters/{name}_head.npy')
      if os.path.exists(f'parameters/{name}_tail.npy'):
        os.remove(f'parameters/{name}_tail.npy')
    for normalize in names_normalize:
      self.assertFalse(os.path.exists(f'parameters/{normalize}_head.npy'))
      self.assertFalse(os.path.exists(f'parameters/{normalize}_tail.npy'))

    for normalize_head in names_normalize:
      for normalize_tail in names_normalize:
        generator = Generator(
            'train',
            batchsize=2,
            normalize_head=normalize_head,
            normalize_tail=normalize_tail,
            return_name=True
        )

        self.assertTrue(generator.on_epoch_end() == None)
        self.assertTrue(len(generator[0]) == 3)
        X, Y, paths = generator[0]
        self.assertTrue(X.shape == (2, 134, 134, 1000))
        self.assertTrue(Y.shape == (2, 1000, 199))
        self.assertTrue(len(paths) == 2)
    for normalize in names_normalize:
      self.assertTrue(os.path.exists(f'parameters/{normalize}_head.npy'))
      self.assertTrue(os.path.exists(f'parameters/{normalize}_tail.npy'))

if __name__ == '__main__':
  unittest.main()