import mongoose from "mongoose";


export const connectDB = async()=>{
try {
  await mongoose.connect("mongodb://localhost:27017/Auth") 
   console.log("Database is connected")
} catch (error) {
    console.log("error in connection to database",error);
}
}
